### code for Phase II Historical Reconstruction: magnitude modeling (method I3-IIA) ###

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
from sklearn.preprocessing import QuantileTransformer

@dataclass
class Config:
    base_dir: Path = Path("outputs/reconstruction_I3")
    input_dir: Path = Path("data/input")
    output_dir: Path = Path("outputs/historical_magnitude_outputs")
    log_dir: Path = Path("logs")
    historical_data_path: Path = Path("combined_dataset.pkl")
    class_dict_path: Path = Path("final_group_days_ALL.pkl")
    variogram_params_path: Path = Path("variogram_parameters_summary.csv")
    metadata_path: Path = Path("merged_rainfall_metadata.csv")
    dem_path: Path = Path("dem.nc")
    trace_threshold: float = 0.0
    min_stations_threshold: int = 4
    min_wet_stations_threshold: int = 3
    max_workers: int = 12
    memory_limit_gb: int = 16
    checkpoint_interval: int = 10
    compression_level: int = 4
    float_dtype: str = "float32"
    int_dtype: str = "int8"

    def __post_init__(self):
        for dir_path in [self.base_dir, self.input_dir, self.output_dir,
                        self.log_dir, self.base_dir/"Phase_II",
                        self.base_dir/"Phase_II"/"Magnitude_Grids",
                        self.base_dir/"Phase_II"/"Uncertainty_Grids",
                        self.base_dir/"Phase_II"/"Metadata"]:
            dir_path.mkdir(parents=True, exist_ok=True)

def setup_logging(config: Config) -> logging.Logger:
    log_file = config.log_dir / f"magnitude_reconstruction_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("MagnitudeReconstruction")
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

    def load_classification(self) -> Dict[str, List[date]]:
        self.logger.info("Loading day classification...")
        with open(self.config.input_dir / self.config.class_dict_path, 'rb') as f:
            class_dict = pickle.load(f)
        converted_dict = {}
        for class_name, dates in class_dict.items():
            converted_dates = []
            for d in dates:
                if isinstance(d, datetime):
                    converted_dates.append(d.date())
                elif isinstance(d, pd.Timestamp):
                    converted_dates.append(d.date())
                elif isinstance(d, str):
                    converted_dates.append(datetime.strptime(d, "%Y-%m-%d").date())
                elif isinstance(d, date):
                    converted_dates.append(d)
            converted_dict[class_name] = converted_dates
        self.logger.info(f"Loaded classification for classes: {list(converted_dict.keys())}")
        return converted_dict

    def load_variogram_parameters(self) -> Dict[str, Dict]:
        self.logger.info("Loading variogram parameters...")
        df_params = pd.read_csv(self.config.input_dir / self.config.variogram_params_path)
        variogram_params = {}
        for _, row in df_params.iterrows():
            category = row['category']
            variogram_params[category] = {
                'variogram_model': row['variogram_model'],
                'range': float(row['length_scale']),
                'sill': float(row['sill']),
                'nugget': float(row['nugget']),
                'nu': float(row['nu']) if 'nu' in row and not pd.isna(row['nu']) else None
            }
        self.logger.info(f"Loaded variogram parameters for {len(variogram_params)} categories")
        return variogram_params

    def load_occurrence_binary_maps(self, year_month: str) -> Optional[xr.Dataset]:
        binary_file = (
            self.config.base_dir /
            "Phase_I" /
            "Binary_Occurrence_Grids" /
            f"binary_occurrence_{year_month}.nc"
        )
        if not binary_file.exists():
            self.logger.warning(f"Binary occurrence file not found: {binary_file}")
            return None
        try:
            ds = xr.open_dataset(binary_file)
            return ds
        except Exception as e:
            self.logger.error(f"Failed to load binary occurrence file {binary_file}: {e}")
            return None

    def load_occurrence_probability_maps(self, year_month: str) -> Optional[xr.Dataset]:
        prob_file = (
            self.config.base_dir /
            "Phase_I" /
            "Occurrence_Probability_Grids" /
            f"occurrence_probability_{year_month}.nc"
        )
        if not prob_file.exists():
            self.logger.warning(f"Probability occurrence file not found: {prob_file}")
            return None
        try:
            ds = xr.open_dataset(prob_file)
            return ds
        except Exception as e:
            self.logger.error(f"Failed to load probability occurrence file {prob_file}: {e}")
            return None

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
        self.logger.info("Creating daily dataframe...")
        merged_data['day'] = merged_data['date'].dt.date
        daily_df = merged_data[['day', 'date', 'station_id', 'Longitude', 'Latitude',
                               'Rain']].copy()
        self.logger.info(f"Daily dataframe created: {daily_df.shape}")
        self.logger.info(f"Date range: {daily_df['day'].min()} to {daily_df['day'].max()}")
        return daily_df

def _matern_variogram_function(params, dist):
    nugget, psill, range_val, nu = params
    model = gs.Matern(dim=2, var=psill, len_scale=range_val, nugget=nugget, nu=nu)
    return model.variogram(dist)

def _translate_gstools_to_pykrige(model_params: Dict) -> Tuple[str, list]:
    if 'nu' in model_params and model_params['nu'] is not None:
        variogram_parameters = [
            model_params["nugget"],
            model_params["sill"] - model_params["nugget"],
            model_params["range"],
            model_params["nu"]
        ]
        return "custom", variogram_parameters
    else:
        variogram_parameters = [
            model_params["nugget"],
            model_params["range"],
            model_params["sill"]
        ]
        return "linear", variogram_parameters

def adaptive_transformation(data: np.ndarray) -> QuantileTransformer:
    transformer = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(10000, len(data)),
        random_state=42,
        subsample=min(20000, len(data))
    )
    data_2d = data.reshape(-1, 1)
    transformer.fit(data_2d)
    return transformer

def transform_variance_to_original_space(
    transformer: QuantileTransformer,
    z_pred_trans: np.ndarray,
    z_var_trans: np.ndarray,
    method: str = 'delta'
) -> np.ndarray:
    if method == 'delta':
        z_pred_flat = z_pred_trans.flatten()
        z_var_flat = z_var_trans.flatten()
        z_var_orig = np.full_like(z_pred_flat, np.nan)
        valid_mask = ~np.isnan(z_pred_flat) & ~np.isnan(z_var_flat)
        z_pred_valid = z_pred_flat[valid_mask]
        z_var_valid = z_var_flat[valid_mask]
        if len(z_pred_valid) == 0:
            return np.full_like(z_pred_trans, np.nan)
        epsilon = 1e-4
        z_pred_plus = z_pred_valid + epsilon
        z_pred_minus = z_pred_valid - epsilon
        z_orig_plus = transformer.inverse_transform(z_pred_plus.reshape(-1, 1)).flatten()
        z_orig_minus = transformer.inverse_transform(z_pred_minus.reshape(-1, 1)).flatten()
        jacobian = (z_orig_plus - z_orig_minus) / (2 * epsilon)
        z_var_orig_valid = (jacobian ** 2) * z_var_valid
        z_var_orig[valid_mask] = z_var_orig_valid
        return z_var_orig.reshape(z_pred_trans.shape)
    elif method == 'monte_carlo':
        n_samples = 100
        z_pred_flat = z_pred_trans.flatten()
        z_var_flat = z_var_trans.flatten()
        valid_mask = ~np.isnan(z_pred_flat) & ~np.isnan(z_var_flat)
        z_pred_valid = z_pred_flat[valid_mask]
        z_var_valid = z_var_flat[valid_mask]
        if len(z_pred_valid) == 0:
            return np.full_like(z_pred_trans, np.nan)
        samples_orig = np.zeros((len(z_pred_valid), n_samples))
        for i in range(n_samples):
            samples_trans = np.random.normal(z_pred_valid, np.sqrt(z_var_valid))
            samples_orig[:, i] = transformer.inverse_transform(
                samples_trans.reshape(-1, 1)
            ).flatten()
        z_var_orig_valid = np.var(samples_orig, axis=1)
        z_var_orig = np.full_like(z_pred_flat, np.nan)
        z_var_orig[valid_mask] = z_var_orig_valid
        return z_var_orig.reshape(z_pred_trans.shape)
    else:
        raise ValueError(f"Unknown method: {method}")

def conditional_kriging_magnitude_ok(
    day: date,
    day_data: pd.DataFrame,
    transformer: QuantileTransformer,
    model_params: Dict,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    land_mask: np.ndarray,
    min_stations: int = 4,
    min_wet_data: int = 3
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    result_grid = np.full((len(grid_y), len(grid_x)), np.nan, dtype=np.float32)
    variance_grid = np.full((len(grid_y), len(grid_x)), np.nan, dtype=np.float32)
    land_cells = land_mask
    sub = day_data[day_data['Rain'] >= 0].copy()
    if len(day_data) < min_stations:
        return result_grid, variance_grid
    if len(sub) < min_wet_data:
        print(f"  ⚠ Day {day}: Only {len(sub)} wet stations - using direct assignment")
        result_grid[land_cells] = 0
        variance_grid[land_cells] = 0
        XX, YY = np.meshgrid(grid_x, grid_y, indexing='xy')
        for _, station in sub.iterrows():
            distances = np.sqrt((XX - station['Longitude'])**2 +
                              (YY - station['Latitude'])**2)
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            if land_mask[min_idx]:
                result_grid[min_idx] = station['Rain']
                variance_grid[min_idx] = 0
        return result_grid, variance_grid
    try:
        x_vals = sub['Longitude'].values
        y_vals = sub['Latitude'].values
        z_vals_raw = sub['Rain'].values
        z_vals_trans = transformer.transform(z_vals_raw.reshape(-1, 1)).flatten()
        model_name, variogram_parameters = _translate_gstools_to_pykrige(model_params)
        variogram_function = None
        if model_name == "custom":
            variogram_function = _matern_variogram_function
        OK = OrdinaryKriging(
            x=x_vals,
            y=y_vals,
            z=z_vals_trans,
            variogram_model=model_name,
            variogram_parameters=variogram_parameters,
            variogram_function=variogram_function,
            exact_values=True,
            pseudo_inv=True,
            verbose=False,
            enable_plotting=False
        )
        z_pred_trans, z_var_trans = OK.execute('masked', xpoints=grid_x, ypoints=grid_y, mask=~land_cells)
        valid_mask = ~np.isnan(z_pred_trans)
        if np.any(valid_mask):
            z_pred_back = transformer.inverse_transform(
                z_pred_trans[valid_mask].reshape(-1, 1)
            ).flatten()
            z_pred_back[z_pred_back < 0] = 0
            z_var_orig = transform_variance_to_original_space(
                transformer,
                z_pred_trans,
                z_var_trans,
                method='monte_carlo'
            )
            temp_pred = np.full_like(z_pred_trans, np.nan)
            temp_var = np.full_like(z_var_trans, np.nan)
            temp_pred[valid_mask] = z_pred_back
            temp_var[valid_mask] = z_var_orig[valid_mask]
            result_grid[land_cells] = temp_pred[land_cells]
            variance_grid[land_cells] = temp_var[land_cells]
        return result_grid, variance_grid
    except Exception as e:
        warnings.warn(f"Kriging failed for {day}: {str(e)}")
        return None, None

def compute_im_product(magnitude_map: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    return magnitude_map * prob_map

class CheckpointSystem:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.checkpoint_file = config.base_dir / "checkpoints" / "magnitude_reconstruction_checkpoint.json"
        self.checkpoint_file.parent.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        current_date: date,
        current_month: str,
        completed_dates: List[date],
        category_results: Dict[str, List[date]],
        metadata: Dict
    ):
        checkpoint_data = {
            'current_date': current_date.isoformat(),
            'current_month': current_month,
            'completed_dates': [d.isoformat() for d in completed_dates],
            'category_results': {
                cls: [d.isoformat() for d in dates]
                for cls, dates in category_results.items()
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
            data['category_results'] = {
                cls: [date.fromisoformat(d) for d in dates]
                for cls, dates in data['category_results'].items()
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
        magnitude_maps: List[np.ndarray],
        uncertainty_maps: List[np.ndarray],
        im_product_maps: List[np.ndarray],
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        land_mask: np.ndarray,
        overwrite: bool = False
    ) -> Tuple[Path, Path]:
        ds = xr.Dataset(
            {
                "rainfall_magnitude": (("time", "y", "x"), np.array(magnitude_maps, dtype=np.float32)),
                "kriging_variance": (("time", "y", "x"), np.array(uncertainty_maps, dtype=np.float32)),
                "IM_product": (("time", "y", "x"), np.array(im_product_maps, dtype=np.float32)),
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
        ds.rainfall_magnitude.attrs = {
            "units": "mm",
            "long_name": "Rainfall magnitude",
            "description": "Conditional kriging magnitude from historical reconstruction",
            "valid_range": [0.0, 1000.0]
        }
        ds.kriging_variance.attrs = {
            "units": "mm²",
            "long_name": "Kriging variance (transformed space)",
            "description": "Kriging variance in transformed space",
            "valid_range": [0.0, 1e6]
        }
        ds.IM_product.attrs = {
            "units": "mm",
            "long_name": "Magnitude multiplied by Occurrence",
            "description": "Rainfall Magnitude as a Product of Occurrence Probability",
            "valid_range": [0.0, 1000.0]
        }
        ds.land_mask.attrs = {
            "units": "binary",
            "long_name": "Land mask",
            "description": "1 for land, 0 for sea",
            "valid_range": [0, 1]
        }
        ds.attrs = {
            "title": f"Historical Rainfall Magnitude Reconstruction - Sicily {year_month}",
            "institution": "UNIPA",
            "source": "Historical gauge observations (1951-2022)",
            "history": f"Created {datetime.now():%Y-%m-%d %H:%M:%S}",
            "conventions": "CF-1.8",
            "reference": "Conditional two-phase rainfall modeling approach",
            "corresponding_data_producer": "Niloufar Beikahmadi",
            "contact": "Niloufar.beikahmadi@gmail.com",
            "github": "https://github.com/Niloufarbeikahmadi",
            "version": "1.0",
            "calendar": "standard",
            "year_month": year_month,
            "start_date": dates[0].isoformat() if dates else "",
            "end_date": dates[-1].isoformat() if dates else "",
            "n_days": len(dates),
            "model_used": "Ordinary Kriging with Matern variogram",
            "trace_threshold": f"{self.config.trace_threshold} mm",
            "min_stations_threshold": self.config.min_stations_threshold
        }
        magnitude_file = (
            self.config.base_dir /
            "Phase_II" /
            "Magnitude_Grids" /
            f"rainfall_magnitude_{year_month}.nc"
        )
        uncertainty_file = (
            self.config.base_dir /
            "Phase_II" /
            "Uncertainty_Grids" /
            f"kriging_variance_{year_month}.nc"
        )
        encoding = {
            'rainfall_magnitude': {
                'zlib': True,
                'complevel': self.config.compression_level,
                'dtype': 'float32',
                '_FillValue': -9999.0
            },
            'kriging_variance': {
                'zlib': True,
                'complevel': self.config.compression_level,
                'dtype': 'float32',
                '_FillValue': -9999.0
            },
            'IM_product': {
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
        ds[['rainfall_magnitude', 'land_mask']].to_netcdf(
            magnitude_file,
            encoding={
                'rainfall_magnitude': encoding['rainfall_magnitude'],
                'land_mask': encoding['land_mask']
            }
        )
        ds[['kriging_variance', 'land_mask']].to_netcdf(
            uncertainty_file,
            encoding={
                'kriging_variance': encoding['kriging_variance'],
                'land_mask': encoding['land_mask']
            }
        )
        im_product_file = (
            self.config.base_dir /
            "Phase_II" /
            "IM_Product_Grids" /
            f"im_product_{year_month}.nc"
        )
        im_product_file.parent.mkdir(parents=True, exist_ok=True)
        ds[['IM_product', 'land_mask']].to_netcdf(
            im_product_file,
            encoding={
                'IM_product': encoding['IM_product'],
                'land_mask': encoding['land_mask']
            }
        )
        self.logger.info(f"Saved monthly files for {year_month}: {magnitude_file.name}, {uncertainty_file.name}, {im_product_file.name}")
        return magnitude_file, uncertainty_file, im_product_file

    def save_metadata(
        self,
        processing_info: Dict,
        model_metadata: Dict,
        quality_flags: Dict
    ):
        metadata_file = self.config.base_dir / "Phase_II" / "Metadata" / "magnitude_reconstruction_metadata.json"
        metadata = {
            "processing_info": processing_info,
            "model_metadata": model_metadata,
            "quality_flags": quality_flags,
            "created": datetime.now().isoformat(),
            "config": {
                "trace_threshold": float(self.config.trace_threshold),
                "min_stations_threshold": int(self.config.min_stations_threshold),
                "min_wet_stations_threshold": int(self.config.min_wet_stations_threshold)
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Metadata saved: {metadata_file}")

class TransformerManager:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.transformers_dir = config.base_dir / "transformers"
        self.transformers_dir.mkdir(exist_ok=True)

    def save_transformer(self, category: str, transformer: QuantileTransformer):
        transformer_file = self.transformers_dir / f"transformer_{category.replace('/', '_')}.pkl"
        try:
            with open(transformer_file, 'wb') as f:
                pickle.dump(transformer, f)
            self.logger.info(f"Saved transformer for {category}")
        except Exception as e:
            self.logger.error(f"Failed to save transformer for {category}: {e}")

    def load_transformer(self, category: str) -> Optional[QuantileTransformer]:
        transformer_file = self.transformers_dir / f"transformer_{category.replace('/', '_')}.pkl"
        if not transformer_file.exists():
            self.logger.warning(f"Transformer file not found for {category}: {transformer_file}")
            return None
        try:
            with open(transformer_file, 'rb') as f:
                transformer = pickle.load(f)
            self.logger.info(f"Loaded transformer for {category}")
            return transformer
        except Exception as e:
            self.logger.error(f"Failed to load transformer for {category}: {e}")
            return None

    def create_and_save_transformer(self, category: str, data: pd.DataFrame) -> Optional[QuantileTransformer]:
        wet_data = data[data['Rain'] >= 0]['Rain'].values
        if len(wet_data) < 10:
            self.logger.warning(f"Insufficient wet data for {category} ({len(wet_data)} points). Using identity transformation.")
        else:
            transformer = adaptive_transformation(wet_data)
        self.save_transformer(category, transformer)
        return transformer

class HistoricalMagnitudeReconstructionPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.data_loader = HistoricalDataLoader(config, self.logger)
        self.checkpoint = CheckpointSystem(config, self.logger)
        self.output_handler = OutputHandler(config, self.logger)
        self.transformer_manager = TransformerManager(config, self.logger)
        self.daily_df = None
        self.class_dict = None
        self.variogram_params = None
        self.dem = None
        self.grid_x = None
        self.grid_y = None
        self.land_mask = None
        self.transformers = {}
        self.processed_dates = []
        self.failed_dates = []
        self.current_month = None
        self.current_month_dates = []
        self.current_month_magnitude_maps = []
        self.current_month_uncertainty_maps = []
        self.current_month_im_product_maps = []

    def load_all_data(self):
        try:
            historical_data = self.data_loader.load_historical_data()
            metadata = self.data_loader.load_metadata()
            merged_data = self.data_loader.merge_data_metadata(historical_data, metadata)
            self.daily_df = self.data_loader.create_daily_dataframe(merged_data)
            self.class_dict = self.data_loader.load_classification()
            self.variogram_params = self.data_loader.load_variogram_parameters()
            self.dem = self.data_loader.load_dem()
            self.grid_x = self.dem.x.values
            self.grid_y = self.dem.y.values
            self.land_mask = (~np.isnan(self.dem['dem'].values)).astype(bool)
            self.logger.info("All data loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def prepare_transformers(self):
        self.logger.info("Preparing transformers for each category...")
        for category, dates in self.class_dict.items():
            if not dates:
                continue
            transformer = self.transformer_manager.load_transformer(category)
            if transformer is None:
                self.logger.info(f"Creating transformer for {category}...")
                cat_data = self.daily_df[self.daily_df['day'].isin(dates)].copy()
                if len(cat_data) > 0:
                    transformer = self.transformer_manager.create_and_save_transformer(category, cat_data)
            if transformer is not None:
                self.transformers[category] = transformer
                self.logger.info(f"Transformer ready for {category}")
            else:
                self.logger.warning(f"Could not prepare transformer for {category}")
        self.logger.info(f"Prepared transformers for {len(self.transformers)} categories")

    def process_day(self, day: date, prob_map: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        day_data = self.daily_df[self.daily_df['day'] == day].copy()
        if len(day_data) < self.config.min_stations_threshold:
            self.logger.warning(f"Insufficient stations for {day}: {len(day_data)}")
            self.failed_dates.append(day)
            return None, None, None
        day_category = None
        for category, dates in self.class_dict.items():
            if day in dates:
                day_category = category
                break
        if not day_category or day_category not in self.variogram_params:
            self.logger.warning(f"No model found for {day}, category: {day_category}")
            self.failed_dates.append(day)
            return None, None, None
        if day_category not in self.transformers:
            self.logger.warning(f"No transformer found for {day}, category: {day_category}")
            self.failed_dates.append(day)
            return None, None, None
        model_params = self.variogram_params[day_category]
        transformer = self.transformers[day_category]
        magnitude_map, uncertainty_map = conditional_kriging_magnitude_ok(
            day=day,
            day_data=day_data,
            transformer=transformer,
            model_params=model_params,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            land_mask=self.land_mask,
            min_stations=self.config.min_stations_threshold,
            min_wet_data=self.config.min_wet_stations_threshold
        )
        if magnitude_map is None or uncertainty_map is None:
            self.logger.warning(f"Kriging failed for {day}")
            self.failed_dates.append(day)
            return None, None, None
        im_product_map = compute_im_product(magnitude_map, prob_map)
        wet_cells = np.sum(~np.isnan(magnitude_map))
        max_rain = np.nanmax(magnitude_map) if wet_cells > 0 else 0
        self.logger.info(f"Processed {day} ({day_category}): "
                        f"{wet_cells} wet cells, max rain: {max_rain:.1f} mm")
        return magnitude_map, uncertainty_map, im_product_map

    def save_current_month(self):
        if not self.current_month_dates:
            return
        try:
            year_month = self.current_month.strftime("%Y-%m")
            self.output_handler.save_monthly_data(
                year_month=year_month,
                dates=self.current_month_dates,
                magnitude_maps=self.current_month_magnitude_maps,
                uncertainty_maps=self.current_month_uncertainty_maps,
                im_product_maps=self.current_month_im_product_maps,
                grid_x=self.grid_x,
                grid_y=self.grid_y,
                land_mask=self.land_mask.astype(np.int8),
                overwrite=True
            )
            self.logger.info(f"Saved month {year_month}: {len(self.current_month_dates)} days")
            self.current_month_dates = []
            self.current_month_magnitude_maps = []
            self.current_month_uncertainty_maps = []
            self.current_month_im_product_maps = []
            return True
        except Exception as e:
            self.logger.error(f"Failed to save month {self.current_month}: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def process_monthly(self):
        all_dates = []
        for dates in self.class_dict.values():
            all_dates.extend(dates)
        all_dates = sorted(set(all_dates))
        monthly_groups = {}
        for day in all_dates:
            month_key = day.strftime("%Y-%m")
            if month_key not in monthly_groups:
                monthly_groups[month_key] = []
            monthly_groups[month_key].append(day)
        processed_months = set()
        for month_key in sorted(monthly_groups.keys()):
            month_dates = monthly_groups[month_key]
            year, month = map(int, month_key.split('-'))
            self.current_month = date(year, month, 1)
            if month_key in processed_months:
                continue
            self.logger.info(f"Processing month: {month_key} ({len(month_dates)} days)")
            prob_ds = self.data_loader.load_occurrence_probability_maps(month_key)
            if prob_ds is None:
                self.logger.error(f"Cannot load probability maps for {month_key}. Skipping month.")
                continue
            for day in tqdm(month_dates, desc=f"Processing {month_key}"):
                if day in self.processed_dates:
                    continue
                try:
                    day_str = day.strftime("%Y-%m-%d")
                    prob_map = prob_ds['prob_map'].sel(time=day_str).values
                    magnitude_map, uncertainty_map, im_product_map = self.process_day(day, prob_map)
                    if magnitude_map is not None and uncertainty_map is not None and im_product_map is not None:
                        self.current_month_dates.append(day)
                        self.current_month_magnitude_maps.append(magnitude_map)
                        self.current_month_uncertainty_maps.append(uncertainty_map)
                        self.current_month_im_product_maps.append(im_product_map)
                        self.processed_dates.append(day)
                    if len(self.processed_dates) % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(day)
                except Exception as e:
                    self.logger.error(f"Error processing {day}: {e}")
                    self.logger.error(traceback.format_exc())
                    self.failed_dates.append(day)
            if self.current_month_dates:
                if self.save_current_month():
                    self.save_checkpoint(
                        self.current_month_dates[-1] if self.current_month_dates else self.current_month
                    )
                    processed_months.add(month_key)
            prob_ds.close()
            gc.collect()

    def save_checkpoint(self, current_date: date):
        category_results = {}
        for category, dates in self.class_dict.items():
            category_dates = [d for d in dates if d in self.processed_dates]
            category_results[category] = category_dates
        self.checkpoint.save_checkpoint(
            current_date=current_date,
            current_month=self.current_month.strftime("%Y-%m") if self.current_month else None,
            completed_dates=self.processed_dates,
            category_results=category_results,
            metadata={
                'variogram_params_loaded': list(self.variogram_params.keys()),
                'transformers_loaded': list(self.transformers.keys()),
                'processed_count': len(self.processed_dates),
                'failed_count': len(self.failed_dates),
                'current_month_size': len(self.current_month_dates)
            }
        )

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("HISTORICAL MAGNITUDE RECONSTRUCTION PIPELINE (MONTHLY SAVING) - WITH IM_PRODUCT")
        self.logger.info("=" * 80)
        try:
            self.load_all_data()
            self.prepare_transformers()
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
                self.process_monthly()
            else:
                self.logger.info("Starting fresh processing")
                self.process_monthly()
            all_dates = []
            for dates in self.class_dict.values():
                all_dates.extend(dates)
            all_dates = sorted(set(all_dates))
            processing_info = {
                'start_date': min(self.processed_dates).isoformat() if self.processed_dates else None,
                'end_date': max(self.processed_dates).isoformat() if self.processed_dates else None,
                'total_days': len(all_dates),
                'successful_days': len(self.processed_dates),
                'failed_days': len(self.failed_dates)
            }
            model_metadata = {}
            for category, params in self.variogram_params.items():
                model_metadata[category] = {
                    'range': params.get('range'),
                    'sill': params.get('sill'),
                    'nugget': params.get('nugget'),
                    'nu': params.get('nu'),
                    'variogram_model': params.get('variogram_model', 'Matern')
                }
            quality_flags = {
                'days_with_insufficient_stations': len(self.failed_dates),
                'min_stations_per_day': self.daily_df.groupby('day').size().min() if not self.daily_df.empty else 0,
                'max_stations_per_day': self.daily_df.groupby('day').size().max() if not self.daily_df.empty else 0,
                'avg_stations_per_day': self.daily_df.groupby('day').size().mean() if not self.daily_df.empty else 0,
                'categories_with_transformers': len(self.transformers)
            }
            self.output_handler.save_metadata(processing_info, model_metadata, quality_flags)
            self.checkpoint.clear_checkpoint()
            self.logger.info("=" * 80)
            self.logger.info("MAGNITUDE RECONSTRUCTION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Processed: {len(self.processed_dates)} days")
            self.logger.info(f"Failed: {len(self.failed_dates)} days")
            self.logger.info(f"Success rate: {len(self.processed_dates)/len(all_dates)*100:.1f}%" if all_dates else "N/A")
            self.logger.info("=" * 80)
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.logger.error(traceback.format_exc())
            if hasattr(self, 'processed_dates') and self.processed_dates:
                last_date = max(self.processed_dates) if self.processed_dates else None
                if last_date:
                    category_results = {}
                    for category, dates in self.class_dict.items():
                        category_dates = [d for d in dates if d in self.processed_dates]
                        category_results[category] = category_dates
                    self.checkpoint.save_checkpoint(
                        current_date=last_date,
                        current_month=self.current_month.strftime("%Y-%m") if self.current_month else None,
                        completed_dates=self.processed_dates,
                        category_results=category_results,
                        metadata={
                            'processed_count': len(self.processed_dates),
                            'failed_count': len(self.failed_dates),
                            'error': str(e)
                        }
                    )
            raise

def main():
    config = Config()
    pipeline = HistoricalMagnitudeReconstructionPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
