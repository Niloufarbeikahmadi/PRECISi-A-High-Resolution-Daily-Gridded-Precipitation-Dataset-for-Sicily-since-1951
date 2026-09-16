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
            model
