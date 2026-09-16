### code for Phase I Historical Reconstruction: applying trained occurrence models to sparse historical data (1951-2022), all methods included###

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import gstools as gs
from pykrige.ok import OrdinaryKriging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import yaml
import pickle
import logging
import warnings
from tqdm import tqdm
from dataclasses import dataclass
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from scipy.spatial import cKDTree

@dataclass
class Config:
    base_dir: Path = Path("outputs/reconstruction_IX")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("outputs/historical_outputs")
    log_dir: Path = Path("logs")
    historical_data_path: Path = Path("combined_dataset.pkl")
    class_dict_path: Path = Path("class_dict_ALL.pkl")
    occ_models_path: Path = Path("class_info_27.11.2025.pkl")
    metadata_path: Path = Path("merged_rainfall_metadata.csv")
    dem_path: Path = Path("dem.nc")
    trace_threshold: float = 0.0
    min_stations_threshold: int = 4
    max_workers: int = 10
    memory_limit_gb: int = 16
    checkpoint_interval: int = 10
    compression_level: int = 4
    float_dtype: str = "float32"
    int_dtype: str = "int8"

    def __post_init__(self):
        for dir_path in [self.base_dir, self.input_dir, self.output_dir,
                        self.log_dir, self.base_dir/"Phase_I",
                        self.base_dir/"Phase_I"/"Occurrence_Probability_Grids",
                        self.base_dir/"Phase_I"/"Binary_Occurrence_Grids",
                        self.base_dir/"Phase_I"/"Metadata"]:
            dir_path.mkdir(parents=True, exist_ok=True)

def setup_logging(config: Config) -> logging.Logger:
    log_file = config.log_dir / f"historical_reconstruction_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("HistoricalReconstruction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger

class HistoricalDataLoader:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_historical_data(self) -> pd.DataFrame:
        self.logger.info("Loading historical rainfall data...")
        df = pd.read_pickle(self.config.input_dir / self.config.historical_data_path)
        self.logger.info(f"Loaded historical data: {df.shape}")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df_long = df.reset_index().melt(
            id_vars=['index'],
            var_name='station_id',
            value_name='Rain'
        )
        df_long.rename(columns={'index': 'date'}, inplace=True)
        df_long['date'] = pd.to_datetime(df_long['date'])
        return df_long

    def load_metadata(self) -> pd.DataFrame:
        self.logger.info("Loading station metadata...")
        metadata = pd.read_csv(self.config.input_dir / self.config.metadata_path)
        required_cols = ['id', 'x', 'y']
        missing = [col for col in required_cols if col not in metadata.columns]
        if missing:
            raise ValueError(f"Missing columns in metadata: {missing}")
        metadata = metadata[['id', 'x', 'y']].rename(
            columns={'id': 'station_id', 'x': 'Longitude', 'y': 'Latitude'}
        )
        self.logger.info(f"Loaded metadata for {len(metadata)} stations")
        return metadata

    def load_classification(self) -> Dict[str, List[datetime]]:
        self.logger.info("Loading day classification...")
        with open(self.config.input_dir / self.config.class_dict_path, 'rb') as f:
            class_dict = pickle.load(f)
        converted_dict = {}
        for class_name, dates in class_dict.items():
            converted_dates = [pd.Timestamp(d).to_pydatetime() for d in dates]
            converted_dict[class_name] = converted_dates
        self.logger.info(f"Loaded classification for classes: {list(converted_dict.keys())}")
        return converted_dict

    def load_occurrence_models(self) -> Dict:
        self.logger.info("Loading occurrence models...")
        with open(self.config.input_dir / self.config.occ_models_path, 'rb') as f:
            occ_models = pickle.load(f)
        self.logger.info(f"Loaded occurrence models for classes: {list(occ_models.keys())}")
        return occ_models

    def load_dem(self) -> xr.Dataset:
        self.logger.info("Loading DEM...")
        dem = xr.open_dataset(self.config.input_dir / self.config.dem_path)
        dem['land_mask'] = (~np.isnan(dem['dem'])).astype(np.int8)
        self.logger.info(f"DEM loaded: shape={dem['dem'].shape}")
        self.logger.info(f"Land cells: {np.sum(dem['land_mask'].values)}")
        self.logger.info(f"Sea cells: {np.sum(dem['land_mask'].values == 0)}")
        return dem

    def merge_data_metadata(self, data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Merging data with metadata...")
        merged = pd.merge(data, metadata, on='station_id', how='left')
        missing_coords = merged[merged['Longitude'].isna() | merged['Latitude'].isna()]
        if len(missing_coords) > 0:
            self.logger.warning(
                f"{len(missing_coords['station_id'].unique())} stations missing coordinates"
            )
            merged = merged.dropna(subset=['Longitude', 'Latitude'])
        self.logger.info(f"Merged data shape: {merged.shape}")
        return merged

    def create_daily_dataframe(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Creating daily dataframe with binary indicators...")
        merged_data['indicator'] = np.where(
            merged_data['Rain'].isna(),
            np.nan,
            (merged_data['Rain'] > self.config.trace_threshold).astype(np.float32)
        )
        merged_data['day'] = merged_data['date'].dt.date
        daily_df = merged_data[['day', 'date', 'station_id', 'Longitude', 'Latitude',
                               'Rain', 'indicator']].copy()
        self.logger.info(f"Daily dataframe created: {daily_df.shape}")
        self.logger.info(f"Date range: {daily_df['day'].min()} to {daily_df['day'].max()}")
        return daily_df

def _matern_variogram_function(params, dist):
    nugget, psill, range_val, nu = params
    model = gs.Matern(dim=2, var=psill, len_scale=range_val, nugget=nugget, nu=nu)
    return model.variogram(dist)

def krige_day_historical(
    day: date,
    day_data: pd.DataFrame,
    model_name: str,
    model_params: Dict,
    grid_lons: np.ndarray,
    grid_lats: np.ndarray,
    land_mask: np.ndarray,
    min_stations: int = 3
) -> Optional[np.ndarray]:
    day_data = day_data.dropna(subset=['indicator'])
    if len(day_data) < min_stations:
        prob_map = np.full((len(grid_lats), len(grid_lons)), np.nan, dtype=np.float32)
        return prob_map
    if day_data['indicator'].nunique() == 1:
        prob_map = np.full((len(grid_lats), len(grid_lons)),
                          day_data['indicator'].iloc[0],
                          dtype=np.float32)
        prob_map[~land_mask] = np.nan
        return prob_map
    try:
        variogram_function = None
        if model_name == "custom" and "nu" in model_params:
            variogram_function = _matern_variogram_function
            variogram_parameters = [
                model_params["nugget"],
                model_params["sill"] - model_params["nugget"],
                model_params["range"],
                model_params["nu"]
            ]
        else:
            variogram_parameters = [
                model_params["nugget"],
                model_params["range"],
                model_params["sill"]
            ]
        OK = OrdinaryKriging(
            x=day_data['Longitude'].values,
            y=day_data['Latitude'].values,
            z=day_data['indicator'].values.astype(float),
            variogram_model=model_name,
            variogram_parameters=variogram_parameters,
            variogram_function=variogram_function,
            exact_values=True,
            pseudo_inv=True
        )
        z, _ = OK.execute('masked', grid_lons, grid_lats, mask=~land_mask)
        z[~land_mask] = np.nan
        return z
    except Exception as e:
        warnings.warn(f"Kriging failed for {day}: {e}")
        return None

def create_binary_map_historical(
    prob_map: np.ndarray,
    cutoff_threshold: float
) -> np.ndarray:
    binary_map = np.full_like(prob_map, np.nan, dtype=np.float32)
    valid_mask = ~np.isnan(prob_map)
    binary_map[valid_mask] = (prob_map[valid_mask] > cutoff_threshold).astype(np.int8)
    return binary_map

class CheckpointSystem:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.checkpoint_file = config.base_dir / "checkpoints" / "historical_reconstruction_checkpoint.json"
        self.checkpoint_file.parent.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        current_date: date,
        current_month: str,
        completed_dates: List[date],
        class_results: Dict[str, List[date]],
        metadata: Dict
    ):
        checkpoint_data = {
            'current_date': current_date.isoformat(),
            'current_month': current_month,
            'completed_dates': [d.isoformat() for d in completed_dates],
            'class_results': {
                cls: [d.isoformat() for d in dates]
                for cls, dates in class_results.items()
            },
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'trace_threshold': self.config.trace_threshold,
                'min_stations': self.config.min_stations_threshold
            }
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            self.logger.info(f"Checkpoint saved: {current_date}, month: {current_month}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[Dict]:
        if not self.checkpoint_file.exists():
            return None
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            data['current_date'] = date.fromisoformat(data['current_date'])
            data['completed_dates'] = [date.fromisoformat(d) for d in data['completed_dates']]
            data['class_results'] = {
                cls: [date.fromisoformat(d) for d in dates]
                for cls, dates in data['class_results'].items()
            }
            self.logger.info(f"Checkpoint loaded: {data['current_date']}, month: {data.get('current_month', 'N/A')}")
            self.logger.info(f"Already completed: {len(data['completed_dates'])} days")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def clear_checkpoint(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            self.logger.info("Checkpoint cleared")

class OutputHandler:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def save_monthly_data(
        self,
        year_month: str,
        dates: List[date],
        prob_maps: List[np.ndarray],
        binary_maps: List[np.ndarray],
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        land_mask: np.ndarray,
        overwrite: bool = False
    ) -> Tuple[Path, Path]:
        ds = xr.Dataset(
            {
                "prob_map": (("time", "y", "x"), np.array(prob_maps, dtype=np.float32)),
                "binary_map": (("time", "y", "x"), np.array(binary_maps, dtype=np.float32)),
                "land_mask": (("y", "x"), land_mask.astype(np.int8))
            },
            coords={
                "time": pd.to_datetime(dates),
                "x": grid_x,
                "y": grid_y
            }
        )
        ds.x.attrs = {"units": "meters", "crs": "EPSG:32633"}
        ds.y.attrs = {"units": "meters", "crs": "EPSG:32633"}
        ds.time.attrs = {"long_name": "Time"}
        ds.prob_map.attrs = {
            "units": "probability",
            "long_name": "Probability of rainfall occurrence",
            "description": "Indicator kriging probability from historical reconstruction",
            "valid_range": [0.0, 1.0]
        }
        ds.binary_map.attrs = {
            "units": "binary",
            "long_name": "Binary rainfall occurrence",
            "description": "Binary rain (1) / no rain (0) map",
            "valid_range": [0, 1]
        }
        ds.land_mask.attrs = {
            "units": "binary",
            "long_name": "Land mask",
            "description": "1 for land, 0 for sea",
            "valid_range": [0, 1]
        }
        ds.attrs = {
            "title": f"Historical Rainfall Occurrence Reconstruction - Sicily {year_month}",
            "institution": "UNIPA",
            "source": "Historical gauge observations (1951-2022)",
            "history": f"Created {datetime.now():%Y-%m-%d %H:%M:%S}",
            "conventions": "CF-1.8",
            "reference": "Conditional two-phase rainfall modeling approach",
            "contact": "Your Contact",
            "version": "1.0",
            "calendar": "standard",
            "year_month": year_month,
            "start_date": dates[0].isoformat() if dates else "",
            "end_date": dates[-1].isoformat() if dates else "",
            "n_days": len(dates)
        }
        prob_file = (
            self.config.base_dir /
            "Phase_I" /
            "Occurrence_Probability_Grids" /
            f"occurrence_probability_{year_month}.nc"
        )
        binary_file = (
            self.config.base_dir /
            "Phase_I" /
            "Binary_Occurrence_Grids" /
            f"binary_occurrence_{year_month}.nc"
        )
        encoding = {
            'prob_map': {
                'zlib': True,
                'complevel': self.config.compression_level,
                'dtype': 'float32',
                '_FillValue': -9999.0
            },
            'binary_map': {
                'zlib': True,
                'complevel': self.config.compression_level,
                'dtype': 'float32',
                '_FillValue': -9999.0
            },
            'land_mask': {
                'zlib': True,
                'complevel': self.config.compression_level,
                'dtype': 'int8',
                '_FillValue': -99
            }
        }
        ds.time.encoding = {
            'units': 'days since 1950-01-01 00:00:00',
            'calendar': 'standard'
        }
        ds.to_netcdf(prob_file, encoding=encoding)
        ds_binary = xr.Dataset(
            {
                "binary_map": (("time", "y", "x"), np.array(binary_maps, dtype=np.float32)),
                "land_mask": (("y", "x"), land_mask.astype(np.int8))
            },
            coords={
                "time": pd.to_datetime(dates),
                "x": grid_x,
                "y": grid_y
            }
        )
        ds_binary.x.attrs = ds.x.attrs.copy()
        ds_binary.y.attrs = ds.y.attrs.copy()
        ds_binary.time.attrs = ds.time.attrs.copy()
        ds_binary.binary_map.attrs = ds.binary_map.attrs.copy()
        ds_binary.land_mask.attrs = ds.land_mask.attrs.copy()
        ds_binary.attrs = ds.attrs.copy()
        ds_binary.attrs["title"] = f"Historical Binary Rainfall Occurrence - Sicily {year_month}"
        ds_binary.time.encoding = ds.time.encoding.copy()
        binary_encoding = {
            'binary_map': encoding['binary_map'].copy(),
            'land_mask': encoding['land_mask'].copy()
        }
        ds_binary.to_netcdf(binary_file, encoding=binary_encoding)
        self.logger.info(f"Saved monthly files for {year_month}: {prob_file.name}, {binary_file.name}")
        return prob_file, binary_file

    def save_metadata(
        self,
        processing_info: Dict,
        model_metadata: Dict,
        quality_flags: Dict
    ):
        metadata_file = self.config.base_dir / "Phase_I" / "Metadata" / "reconstruction_metadata.json"
        metadata = {
            "processing_info": processing_info,
            "model_metadata": model_metadata,
            "quality_flags": quality_flags,
            "created": datetime.now().isoformat(),
            "config": {
                "trace_threshold": float(self.config.trace_threshold),
                "min_stations_threshold": int(self.config.min_stations_threshold)
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Metadata saved: {metadata_file}")

class HistoricalReconstructionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.data_loader = HistoricalDataLoader(config, self.logger)
        self.checkpoint = CheckpointSystem(config, self.logger)
        self.output_handler = OutputHandler(config, self.logger)
        self.daily_df = None
        self.class_dict = None
        self.occ_models = None
        self.dem = None
        self.grid_lons = None
        self.grid_lats = None
        self.land_mask = None
        self.processed_dates = []
        self.failed_dates = []
        self.current_month = None
        self.current_month_dates = []
        self.current_month_prob_maps = []
        self.current_month_binary_maps = []

    def load_all_data(self):
        try:
            historical_data = self.data_loader.load_historical_data()
            metadata = self.data_loader.load_metadata()
            merged_data = self.data_loader.merge_data_metadata(historical_data, metadata)
            self.daily_df = self.data_loader.create_daily_dataframe(merged_data)
            self.class_dict = self.data_loader.load_classification()
            self.occ_models = self.data_loader.load_occurrence_models()
            self.dem = self.data_loader.load_dem()
            self.grid_lons = self.dem.x.values
            self.grid_lats = self.dem.y.values
            self.land_mask = (~np.isnan(self.dem['dem'].values)).astype(bool)
            self.logger.info("All data loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def prepare_model_info(self) -> Dict:
        model_info = {}
        manual_cutoffs = {
            'F0-25': 0.62,
            'F25-75': 0.47,
            'F75-100': 0.30
        }
        for class_name, model_data in self.occ_models.items():
            best_model = model_data.get('best_model', {})
            model_params = best_model.get('model_params', {})
            model_info[class_name] = {
                'pykrige_model': best_model.get('pykrige_model', 'custom'),
                'model_params': model_params,
                'cutoff_threshold': float(model_data.get('cutoff_threshold', 0.5)),
                'days_in_training': model_data.get('days_count', 0)
            }
            if class_name in manual_cutoffs:
                model_info[class_name]['cutoff_threshold'] = manual_cutoffs[class_name]
            self.logger.info(f"Model for {class_name}: "
                           f"range={model_params.get('range', 'N/A'):.0f}, "
                           f"cutoff={model_info[class_name]['cutoff_threshold']:.3f}")
        return model_info

    def process_day(self, day: date, model_info: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        day_data = self.daily_df[self.daily_df['day'] == day].copy()
        if len(day_data) < self.config.min_stations_threshold:
            self.logger.warning(f"Insufficient stations for {day}: {len(day_data)}")
            self.failed_dates.append(day)
            return None, None
        day_class = None
        day_datetime = datetime.combine(day, datetime.min.time())
        for class_name, dates in self.class_dict.items():
            if day_datetime in dates:
                day_class = class_name
                break
        if not day_class or day_class not in model_info:
            self.logger.warning(f"No model found for {day}, class: {day_class}")
            self.failed_dates.append(day)
            return None, None
        class_info = model_info[day_class]
        prob_map = krige_day_historical(
            day=day,
            day_data=day_data,
            model_name=class_info['pykrige_model'],
            model_params=class_info['model_params'],
            grid_lons=self.grid_lons,
            grid_lats=self.grid_lats,
            land_mask=self.land_mask,
            min_stations=self.config.min_stations_threshold
        )
        if prob_map is None:
            self.logger.warning(f"Kriging failed for {day}")
            self.failed_dates.append(day)
            return None, None
        binary_map = create_binary_map_historical(
            prob_map=prob_map,
            cutoff_threshold=class_info['cutoff_threshold']
        )
        self.logger.info(f"Processed {day} ({day_class}): "
                        f"{np.nansum(binary_map == 1)} wet cells")
        return prob_map, binary_map

    def save_current_month(self):
        if not self.current_month_dates:
            return
        try:
            year_month = self.current_month.strftime("%Y-%m")
            self.output_handler.save_monthly_data(
                year_month=year_month,
                dates=self.current_month_dates,
                prob_maps=self.current_month_prob_maps,
                binary_maps=self.current_month_binary_maps,
                grid_x=self.grid_lons,
                grid_y=self.grid_lats,
                land_mask=self.land_mask.astype(np.int8),
                overwrite=True
            )
            self.logger.info(f"Saved month {year_month}: {len(self.current_month_dates)} days")
            self.current_month_dates = []
            self.current_month_prob_maps = []
            self.current_month_binary_maps = []
            return True
        except Exception as e:
            self.logger.error(f"Failed to save month {self.current_month}: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def process_monthly(self, model_info: Dict):
        all_dates = sorted(self.daily_df['day'].unique())
        monthly_groups = {}
        for day in all_dates:
            month_key = day.strftime("%Y-%m")
            if month_key not in monthly_groups:
                monthly_groups[month_key] = []
            monthly_groups[month_key].append(day)
        for month_key in sorted(monthly_groups.keys()):
            month_dates = monthly_groups[month_key]
            year, month = map(int, month_key.split('-'))
            self.current_month = date(year, month, 1)
            self.logger.info(f"Processing month: {month_key} ({len(month_dates)} days)")
            for day in tqdm(month_dates, desc=f"Processing {month_key}"):
                if day in self.processed_dates:
                    continue
                try:
                    prob_map, binary_map = self.process_day(day, model_info)
                    if prob_map is not None and binary_map is not None:
                        self.current_month_dates.append(day)
                        self.current_month_prob_maps.append(prob_map)
                        self.current_month_binary_maps.append(binary_map)
                        self.processed_dates.append(day)
                    if len(self.processed_dates) % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(day, model_info)
                except Exception as e:
                    self.logger.error(f"Error processing {day}: {e}")
                    self.logger.error(traceback.format_exc())
                    self.failed_dates.append(day)
            if self.current_month_dates:
                if self.save_current_month():
                    self.save_checkpoint(
                        self.current_month_dates[-1] if self.current_month_dates else self.current_month,
                        model_info
                    )
            gc.collect()

    def save_checkpoint(self, current_date: date, model_info: Dict):
        class_results = {}
        for class_name, dates in self.class_dict.items():
            class_dates = [d for d in dates if datetime.combine(d, datetime.min.time()) in self.processed_dates]
            class_results[class_name] = class_dates
        self.checkpoint.save_checkpoint(
            current_date=current_date,
            current_month=self.current_month.strftime("%Y-%m") if self.current_month else None,
            completed_dates=self.processed_dates,
            class_results=class_results,
            metadata={
                'model_info': model_info,
                'processed_count': len(self.processed_dates),
                'failed_count': len(self.failed_dates),
                'current_month_size': len(self.current_month_dates)
            }
        )

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("HISTORICAL OCCURRENCE RECONSTRUCTION PIPELINE (MONTHLY SAVING)")
        self.logger.info("=" * 80)
        try:
            self.load_all_data()
            model_info = self.prepare_model_info()
            checkpoint_data = self.checkpoint.load_checkpoint()
            if checkpoint_data:
                self.processed_dates = checkpoint_data['completed_dates']
                start_date = checkpoint_data['current_date']
                current_month = checkpoint_data.get('current_month')
                if current_month:
                    year, month = map(int, current_month.split('-'))
                    self.current_month = date(year, month, 1)
                self.logger.info(f"Resuming from checkpoint: {start_date}")
                self.logger.info(f"Resuming from month: {current_month}")
                self.logger.info(f"Already processed: {len(self.processed_dates)} days")
                all_dates = sorted(self.daily_df['day'].unique())
                remaining_dates = [d for d in all_dates if d > start_date]
                self.logger.info(f"Remaining days: {len(remaining_dates)}")
                self.process_monthly(model_info)
            else:
                self.logger.info("Starting fresh processing")
                self.process_monthly(model_info)
            all_dates = sorted(self.daily_df['day'].unique())
            processing_info = {
                'start_date': min(self.processed_dates).isoformat() if self.processed_dates else None,
                'end_date': max(self.processed_dates).isoformat() if self.processed_dates else None,
                'total_days': len(all_dates),
                'successful_days': len(self.processed_dates),
                'failed_days': len(self.failed_dates)
            }
            model_metadata = {}
            for class_name, info in model_info.items():
                model_metadata[class_name] = {
                    'cutoff_threshold': info['cutoff_threshold'],
                    'range': info['model_params'].get('range'),
                    'sill': info['model_params'].get('sill'),
                    'nugget': info['model_params'].get('nugget'),
                    'nu': info['model_params'].get('nu')
                }
            quality_flags = {
                'days_with_insufficient_stations': len(self.failed_dates),
                'min_stations_per_day': self.daily_df.groupby('day').size().min(),
                'max_stations_per_day': self.daily_df.groupby('day').size().max(),
                'avg_stations_per_day': self.daily_df.groupby('day').size().mean()
            }
            self.output_handler.save_metadata(processing_info, model_metadata, quality_flags)
            self.checkpoint.clear_checkpoint()
            self.logger.info("=" * 80)
            self.logger.info("RECONSTRUCTION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Processed: {len(self.processed_dates)} days")
            self.logger.info(f"Failed: {len(self.failed_dates)} days")
            self.logger.info(f"Success rate: {len(self.processed_dates)/len(all_dates)*100:.1f}%")
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            if hasattr(self, 'processed_dates') and self.processed_dates:
                last_date = max(self.processed_dates) if self.processed_dates else None
                if last_date:
                    class_results = {}
                    for class_name, dates in self.class_dict.items():
                        class_dates = [d for d in dates if datetime.combine(d, datetime.min.time()) in self.processed_dates]
                        class_results[class_name] = class_dates
                    self.checkpoint.save_checkpoint(
                        current_date=last_date,
                        current_month=self.current_month.strftime("%Y-%m") if self.current_month else None,
                        completed_dates=self.processed_dates,
                        class_results=class_results,
                        metadata={
                            'model_info': model_info if 'model_info' in locals() else {},
                            'processed_count': len(self.processed_dates),
                            'failed_count': len(self.failed_dates),
                            'error': str(e)
                        }
                    )
            raise

def main():
    config = Config()
    pipeline = HistoricalReconstructionPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
