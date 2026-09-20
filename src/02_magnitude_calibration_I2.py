### code for Phase II: rain magnitude modeling, method I2-IIA ###

import numpy as np
import pandas as pd
import gstools as gs
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.stats import shapiro, probplot, pearsonr, boxcox
from scipy.spatial import cKDTree
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr
from datetime import date,datetime
from typing import Dict, List, Any, Tuple, Optional
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Transformer
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.model_selection import KFold
import dill, os, pickle, warnings, json, gc
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.metrics import mean_squared_error
import hydroeval
import dill,pprint, sys, os
from pathlib import Path
import logging

def ensure_progress_dir():
    progress_dir = "progress_tracking"
    os.makedirs(progress_dir, exist_ok=True)
    return progress_dir

def load_progress(workflow_type: str, category: str) -> Dict:
    progress_dir = ensure_progress_dir()
    progress_file = os.path.join(progress_dir, f"progress_{workflow_type}_{category}.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
            if "completed_days" in data:
                data["completed_days"] = [date.fromisoformat(d) for d in data["completed_days"]]
            print(f"  Loaded progress: {len(data.get('completed_days', []))} days completed")
            return data
        except Exception as e:
            print(f"  Error loading progress file: {e}. Starting fresh.")
    return {"completed_days": [], "current_category": None}

def save_progress(workflow_type: str, category: str, completed_days: List[date]):
    progress_dir = ensure_progress_dir()
    progress_file = os.path.join(progress_dir, f"progress_{workflow_type}_{category}.json")
    completed_days_str = [d.isoformat() for d in completed_days]
    progress_data = {
        "completed_days": completed_days_str,
        "category": category,
        "workflow_type": workflow_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_completed": len(completed_days)
    }
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        print(f"  Progress saved: {len(completed_days)} days completed")
    except Exception as e:
        warnings.warn(f"Could not save progress: {e}")

def save_enhanced_report(report: str, category: str, workflow_type: str, save_path: str):
    report_file = os.path.join(save_path, f"model_report_{workflow_type}_{category}_30.11.2025.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

def cleanup_progress(workflow_type: str, category: str):
    progress_dir = ensure_progress_dir()
    progress_file = os.path.join(progress_dir, f"progress_{workflow_type}_{category}.json")
    if os.path.exists(progress_file):
        try:
            os.remove(progress_file)
            print(f"  Cleaned up progress file for {category}")
        except Exception as e:
            warnings.warn(f"Could not remove progress file: {e}")

def recover_interrupted_run(workflow_type="OK"):
    progress_dir = ensure_progress_dir()
    print(f"\nRecovery Status for {workflow_type}:")
    print("=" * 50)
    progress_files = [f for f in os.listdir(progress_dir) if f.startswith(f"progress_{workflow_type}")]
    if not progress_files:
        print("No interrupted runs found.")
        return
    for progress_file in progress_files:
        category = progress_file.replace(f"progress_{workflow_type}_", "").replace(".json", "")
        try:
            with open(os.path.join(progress_dir, progress_file), 'r') as f:
                data = json.load(f)
            completed = len(data.get('completed_days', []))
            timestamp = data.get('timestamp', 'Unknown')
            print(f"{category}:")
            print(f"   Completed: {completed} days")
            print(f"   Last save: {timestamp}")
            print(f"   File: {progress_file}")
            print()
        except Exception as e:
            print(f"Error reading {progress_file}: {e}")

def _matern_variogram_function(params, dist):
    nugget, psill, range_val, nu = params
    model = gs.Matern(dim=2, var=psill, len_scale=range_val, nugget=nugget, nu=nu)
    return model.variogram(dist)

def _translate_gstools_to_pykrige(gs_model: gs.CovModel) -> Tuple[str, dict]:
    model_name = gs_model.__class__.__name__.lower()
    if model_name == "matern":
        params = {
            "sill": float(gs_model.var),
            "range": float(gs_model.len_scale),
            "nugget": float(gs_model.nugget),
            "nu": float(gs_model.nu)
        }
        return "custom", params
    else:
        params = {
            "sill": float(gs_model.var),
            "range": float(gs_model.len_scale),
            "nugget": float(gs_model.nugget),
        }
        return model_name, params

def setup_simple_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/workflow_simple24.11.2025.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_simple_logging()

def diagnose_category_quality(category_df: pd.DataFrame, days: List[date],
                              category_name: str) -> Dict[str, Any]:
    diagnostics = {
        'category': category_name,
        'total_days': len(days),
        'stations_per_day': [],
        'spatial_coverage': [],
        'value_statistics': {},
        'recommendations': []
    }
    for day in days:
        day_data = category_df[(category_df['day'] == day) & (category_df['Rain'] > 0)]
        n_stations = len(day_data)
        diagnostics['stations_per_day'].append(n_stations)
        if n_stations >= 2:
            x_range = day_data['Longitude'].max() - day_data['Longitude'].min()
            y_range = day_data['Latitude'].max() - day_data['Latitude'].min()
            spatial_extent = np.sqrt(x_range**2 + y_range**2)
            diagnostics['spatial_coverage'].append(spatial_extent)
    if diagnostics['stations_per_day']:
        diagnostics['avg_stations'] = np.mean(diagnostics['stations_per_day'])
        diagnostics['min_stations'] = np.min(diagnostics['stations_per_day'])
        diagnostics['days_with_lt_3_stations'] = sum(1 for x in diagnostics['stations_per_day'] if x < 3)
        diagnostics['days_with_lt_5_stations'] = sum(1 for x in diagnostics['stations_per_day'] if x < 5)
    else:
        diagnostics['avg_stations'] = 0
        diagnostics['min_stations'] = 0
        diagnostics['days_with_lt_3_stations'] = len(days)
        diagnostics['days_with_lt_5_stations'] = len(days)
    all_rain = category_df[category_df['Rain'] >0]['Rain'].values
    if len(all_rain) > 0:
        diagnostics['value_statistics'] = {
            'count': len(all_rain),
            'mean': np.mean(all_rain),
            'median': np.median(all_rain),
            'std': np.std(all_rain),
            'cv': np.std(all_rain) / np.mean(all_rain) if np.mean(all_rain) > 0 else 0,
            'min': np.min(all_rain),
            'max': np.max(all_rain),
            'q25': np.percentile(all_rain, 25),
            'q75': np.percentile(all_rain, 75)
        }
    else:
        diagnostics['value_statistics'] = {'count': 0}
    if diagnostics['avg_stations'] < 5:
        diagnostics['recommendations'].append("LOW_STATION_DENSITY: Avg stations/day < 5. Spatial K-Fold CV triggered.")
    if diagnostics['days_with_lt_3_stations'] / len(days) > 0.3:
        diagnostics['recommendations'].append("INSUFFICIENT_DAILY_DATA: >30% of days have <3 wet stations.")
    if diagnostics['value_statistics'].get('cv', 0) > 2.0:
        diagnostics['recommendations'].append("HIGH_VARIABILITY: CV > 2.0. Data is highly skewed.")
    if diagnostics['value_statistics'].get('mean', 0) < 2.0 and diagnostics['value_statistics']['count'] > 0:
        diagnostics['recommendations'].append("LOW_VALUES: Mean rainfall < 2.0mm. Transformation may be sensitive.")
    if not diagnostics['spatial_coverage'] and diagnostics['value_statistics']['count'] > 0:
        diagnostics['recommendations'].append("NO_SPATial_DATA: No days with >1 station. Kriging impossible.")
    return diagnostics

def handle_problematic_categories(category: str, diagnostics: Dict) -> Optional[str]:
    if category == "F25-75_Heavier":
        if diagnostics['value_statistics'].get('max', 0) > 100:
            warnings.warn(f"INFO [{category}]: Extreme values detected. Outlier capping recommended.")
        if diagnostics['value_statistics'].get('cv', 0) > 3:
            return "USE_ROBUST_VARIOGRAM"
    if 'F75-100' in category and 'Lighter' in category and diagnostics['avg_stations'] < 3:
        warnings.warn(f"CRITICAL [{category}]: Avg stations < 3. Recommending MERGE.")
        return "MERGE_WITH_MODERATE"
    if 'F75-100' in category:
        if diagnostics['spatial_coverage'] and np.mean(diagnostics['spatial_coverage']) < 50000:
            warnings.warn(f"INFO [{category}]: Very localized rainfall. Spatial structure may be weak.")
            return "INSUFFICIENT_SPATIAL_STRUCTURE"
    return "OK"

def detect_and_handle_outliers(category_df: pd.DataFrame, diagnostics: Dict, days: List[date]) -> pd.DataFrame:
    df_clean = category_df.copy()
    return df_clean

def adaptive_transformation(data: np.ndarray, diagnostics: Dict) -> Tuple[Optional[Any], Optional[str]]:
    n_samples = len(data)
    category_name = diagnostics.get('category', 'Unknown')
    fig_dir = "transformation_plots"
    os.makedirs(fig_dir, exist_ok=True)
    try:
        n_quantiles = min(n_samples, 10000)
        print(f"  Quantile Transformation: {n_samples} samples -> {n_quantiles} quantiles")
        transformer = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=n_quantiles,
            random_state=42,
            subsample=min(20000, n_samples)
        )
        data_2d = data.reshape(-1, 1)
        transformed_data = transformer.fit_transform(data_2d).flatten()
        if n_samples <= 5000:
            _, p_val = shapiro(transformed_data)
        else:
            sample_size = min(5000, n_samples)
            indices = np.linspace(0, n_samples-1, sample_size, dtype=int)
            _, p_val = shapiro(transformed_data[indices])
        create_transformation_plots(data, transformed_data, category_name, p_val, fig_dir)
        print(f"  Quantile transformation complete: n_quantiles={n_quantiles}, p-value={p_val:.4f}")
        return transformer, 'quantile'
    except Exception as e:
        warnings.warn(f"Quantile transformation failed for {category_name}: {e}")
        try:
            identity_transformer = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x, validate=True)
            print("  ! Using identity transformation as fallback")
            return identity_transformer, 'quantile'
        except:
            pass
    return None, None

def create_transformation_plots(raw_data: np.ndarray, transformed_data: np.ndarray, 
                              category: str, p_value: float, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Transformation Analysis: {category}\nShapiro-Wilk p-value: {p_value:.4f}', 
                 fontsize=14, fontweight='bold')
    axes[0].hist(raw_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Rainfall (mm)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Raw Data Distribution')
    axes[0].grid(True, alpha=0.3)
    stats_text = f'Mean: {np.mean(raw_data):.2f}\nStd: {np.std(raw_data):.2f}\nMax: {np.max(raw_data):.2f}'
    axes[0].text(0.95, 0.95, stats_text, transform=axes[0].transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[1].hist(transformed_data, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Transformed Values')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('After Quantile Transformation')
    axes[1].grid(True, alpha=0.3)
    stats_text_trans = f'Mean: {np.mean(transformed_data):.2f}\nStd: {np.std(transformed_data):.2f}'
    axes[1].text(0.95, 0.95, stats_text_trans, transform=axes[1].transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    probplot(transformed_data, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{save_dir}/transformation_{category.replace('/', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved transformation plot: {filename}")

def get_adaptive_variogram_bounds(category_name: str, diagnostics: Dict,
                                  bin_centers: np.ndarray, gamma: np.ndarray) -> Dict:
    max_gamma = np.nanmax(gamma)
    if not np.isfinite(max_gamma) or max_gamma == 0:
        max_gamma = 1.0
    bounds = {
        'len_scale': [1000.0, 150000.0],
        'nu': [0.2, 5.0],
        'nugget': [0, max_gamma * 0.1]
    }
    print(f"  Using universal bounds: len_scale={bounds['len_scale']}, nu={bounds['nu']}, nugget={bounds['nugget']}")
    return bounds

def _calculate_cv_scores(obs_vals_orig, pred_vals_orig, obs_vals_trans, pred_vals_trans, variance_vals) -> Dict:
    valid_mask_orig = (np.isfinite(pred_vals_orig) & np.isfinite(obs_vals_orig))
    valid_mask_trans = (np.isfinite(pred_vals_trans) & 
                        np.isfinite(obs_vals_trans) & 
                        np.isfinite(variance_vals) & 
                        (variance_vals > 1e-9))
    if valid_mask_trans.sum() < 10:
        raise ValueError("Insufficient valid points for CV scores (<10)")
    residuals_trans = obs_vals_trans[valid_mask_trans] - pred_vals_trans[valid_mask_trans]
    standard_errors = np.sqrt(variance_vals[valid_mask_trans])
    standardized_residuals = residuals_trans / standard_errors
    s1 = np.nanmean(standardized_residuals)
    s2 = np.nanmean(np.square(standardized_residuals))
    obs_clean = obs_vals_orig[valid_mask_orig]
    pred_clean = pred_vals_orig[valid_mask_orig]
    if len(obs_clean) < 10:
        raise ValueError("Insufficient valid original-space points for CV scores (<10)")
    RMSE = float(hydroeval.evaluator(hydroeval.rmse, obs_clean, pred_clean)[0])
    p_bias = float(hydroeval.evaluator(hydroeval.pbias, pred_clean, obs_clean)[0])
    corr, _ = pearsonr(obs_clean, pred_clean)
    KGE = float(hydroeval.evaluator(hydroeval.kge, pred_clean, obs_clean)[0])
    return {
        "s1": float(s1), "s2": float(s2), "rmse": float(RMSE),
        "rel_bias": float(p_bias), "kge": float(KGE), "corr": float(corr)
    }

def daily_loocv(category_df: pd.DataFrame, days: List[date],
                model_name: str, model_params: dict, normalizer: Any,
                use_elevation: bool) -> Dict:
    all_obs_orig, all_pred_orig = [], []
    all_obs_trans, all_pred_trans, all_variances = [], [], []
    min_points_day = 4 if use_elevation else 3
    pbar = tqdm(days, desc=f"  Daily LOOCV", leave=False)
    for day in pbar:
        day_df = category_df[(category_df['day'] == day) & (category_df['Rain'] > 0)]
        if len(day_df) < min_points_day:
            continue
        x = day_df['Longitude'].values
        y = day_df['Latitude'].values
        z_raw = day_df['Rain'].values
        z_trans = day_df['Rain_transformed'].values
        predictions_normalized = np.full_like(z_trans, np.nan)
        variances = np.full_like(z_trans, np.nan)
        variogram_function = None
        variogram_parameters = None
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
        for i in range(len(x)):
            try:
                mask = np.ones(len(x), dtype=bool)
                mask[i] = False
                x_train, y_train, z_train = x[mask], y[mask], z_trans[mask]
                if not use_elevation:
                    ok = OrdinaryKriging(
                        x_train, y_train, z_train,
                        variogram_model=model_name,
                        variogram_parameters=variogram_parameters,
                        variogram_function=variogram_function,
                        verbose=False, enable_plotting=False,
                        pseudo_inv=True, exact_values=False
                    )
                    pred, var = ok.execute('points', [x[i]], [y[i]])
                else:
                    elev = day_df['Elevation'].values
                    elev_train = elev[mask]
                    uk = UniversalKriging(
                        x_train, y_train, z_train,
                        variogram_model=model_name,
                        variogram_parameters=variogram_parameters,
                        variogram_function=variogram_function,
                        drift_terms=['external_Z'],
                        external_drift=elev_train,
                        verbose=False, enable_plotting=False,
                        pseudo_inv=True, exact_values=False
                    )
                    pred, var = uk.execute('points', [x[i]], [y[i]], 
                                           external_drift=np.array([elev[i]]))
                predictions_normalized[i] = pred[0]
                variances[i] = var[0]
            except Exception as e:
                continue
        predictions_original = np.full_like(predictions_normalized, np.nan)
        valid_mask = np.isfinite(predictions_normalized)
        if np.any(valid_mask):
            try:
                predictions_original[valid_mask] = normalizer.inverse_transform(
                    predictions_normalized[valid_mask].reshape(-1, 1)
                ).flatten()
                predictions_original[predictions_original <= 0.2] = 0
            except Exception:
                continue
        all_obs_orig.append(z_raw)
        all_pred_orig.append(predictions_original)
        all_obs_trans.append(z_trans)
        all_pred_trans.append(predictions_normalized)
        all_variances.append(variances)
    if not all_obs_orig:
        raise ValueError("No valid CV results from any day in Daily LOOCV")
    obs_vals_orig = np.concatenate(all_obs_orig)
    pred_vals_orig = np.concatenate(all_pred_orig)
    obs_vals_trans = np.concatenate(all_obs_trans)
    pred_vals_trans = np.concatenate(all_pred_trans)
    variance_vals = np.concatenate(all_variances)
    return _calculate_cv_scores(obs_vals_orig, pred_vals_orig, obs_vals_trans, pred_vals_trans, variance_vals)

def spatial_kfold_cv(category_df: pd.DataFrame, days: List[date], 
                     model_name: str, model_params: dict, normalizer: Any, 
                     use_elevation: bool, n_folds=5) -> Dict:
    all_obs_orig, all_pred_orig = [], []
    all_obs_trans, all_pred_trans, all_variances = [], [], []
    wet_data = category_df[category_df['Rain']> 0].copy()
    if len(wet_data) < 20:
        raise ValueError(f"Insufficient total observations for k-fold CV (need 20, have {len(wet_data)})")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    print(f"  Using Spatial K-Fold CV (n_folds={n_folds}) on {len(wet_data)} total points.")
    for train_idx, test_idx in kf.split(wet_data):
        train_data = wet_data.iloc[train_idx]
        test_data = wet_data.iloc[test_idx]
        x_train = train_data['Longitude'].values
        y_train = train_data['Latitude'].values
        z_train = train_data['Rain_transformed'].values
        x_test = test_data['Longitude'].values
        y_test = test_data['Latitude'].values
        z_test_trans = test_data['Rain_transformed'].values
        z_test_orig = test_data['Rain'].values
        try:
            variogram_function = None
            variogram_parameters = None
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
            if not use_elevation:
                ok = OrdinaryKriging(
                    x_train, y_train, z_train,
                    variogram_model=model_name,
                    variogram_parameters=variogram_parameters,
                    variogram_function=variogram_function,
                    verbose=False, enable_plotting=False,
                    pseudo_inv=True, exact_values=False
                )
                pred_trans, var = ok.execute('points', x_test, y_test)
            else:
                elev_train = train_data['Elevation'].values
                elev_test = test_data['Elevation'].values
                uk = UniversalKriging(
                    x_train, y_train, z_train,
                    variogram_model=model_name,
                    variogram_parameters=variogram_parameters,
                    variogram_function=variogram_function,
                    drift_terms=['external_Z'],
                    external_drift=elev_train,
                    verbose=False, enable_plotting=False,
                    pseudo_inv=True, exact_values=False
                )
                pred_trans, var = uk.execute('points', x_test, y_test, 
                                            external_drift=elev_test)
            pred_orig = np.full_like(pred_trans, np.nan)
            valid_mask = np.isfinite(pred_trans)
            if np.any(valid_mask):
                pred_orig[valid_mask] = normalizer.inverse_transform(
                    pred_trans[valid_mask].reshape(-1, 1)
                ).flatten()
                pred_orig[pred_orig < 0] = 0
            all_obs_trans.append(z_test_trans)
            all_pred_trans.append(pred_trans)
            all_variances.append(var)
            all_obs_orig.append(z_test_orig)
            all_pred_orig.append(pred_orig)
        except Exception as e:
            warnings.warn(f"K-fold iteration failed: {e}")
            continue
    if not all_obs_orig:
        raise ValueError("No valid CV results from any fold in K-Fold CV")
    obs_vals_orig = np.concatenate(all_obs_orig)
    pred_vals_orig = np.concatenate(all_pred_orig)
    obs_vals_trans = np.concatenate(all_obs_trans)
    pred_vals_trans = np.concatenate(all_pred_trans)
    variance_vals = np.concatenate(all_variances)
    return _calculate_cv_scores(obs_vals_orig, pred_vals_orig, obs_vals_trans, pred_vals_trans, variance_vals)

def adaptive_cross_validation(
    category_df: pd.DataFrame,
    days: List[date],
    model_name: str,
    model_params: dict,
    normalizer: Any,
    diagnostics: Dict,
    use_elevation: bool
) -> Dict:
    avg_stations = diagnostics['avg_stations']
    if avg_stations < 5:
        print("  Triggering Spatial K-Fold CV (Avg stations < 5)")
        try:
            return spatial_kfold_cv(category_df, days, model_name, model_params, normalizer, use_elevation)
        except Exception as e_kfold:
            warnings.warn(f"Spatial K-Fold CV failed ({e_kfold}). Attempting Daily LOOCV as fallback.")
            try:
                return daily_loocv(category_df, days, model_name, model_params, normalizer, use_elevation)
            except Exception as e_daily:
                raise ValueError(f"Both Spatial K-Fold and Daily LOOCV failed. K-Fold: {e_kfold}, Daily: {e_daily}")
    else:
        print("  Triggering Daily LOOCV (Avg stations >= 5)")
        try:
            return daily_loocv(category_df, days, model_name, model_params, normalizer, use_elevation)
        except Exception as e_daily:
            warnings.warn(f"Daily LOOCV failed ({e_daily}). Attempting Spatial K-Fold as fallback.")
            try:
                return spatial_kfold_cv(category_df, days, model_name, model_params, normalizer, use_elevation)
            except Exception as e_kfold:
                raise ValueError(f"Both Daily LOOCV and Spatial K-Fold failed. Daily: {e_daily}, K-Fold: {e_kfold}")

def enhanced_model_reporting(category: str, diagnostics: Dict,
                             cv_scores: Dict, variogram_params: Dict,
                             trans_name: str) -> str:
    report = f"""
    {'='*78}
    CATEGORY: {category:<66}
    {'='*78}
    DATA QUALITY METRICS                   Transformation: {trans_name:<19}
      - Total Days: {diagnostics['total_days']:<28} - Mean Rainfall: {diagnostics['value_statistics'].get('mean', 0):<6.2f} mm
      - Avg Stations/Day: {diagnostics['avg_stations']:<22.2f} - CV: {diagnostics['value_statistics'].get('cv', 0):<12.2f}
      - Days with <3 stations: {diagnostics['days_with_lt_3_stations']:<19} - Total Wet Points: {diagnostics['value_statistics'].get('count', 0):<6}
    {'='*78}
    VARIOGRAM PARAMETERS (Matern)
      - Length Scale: {variogram_params.get('range', 0):<26.0f} m - Sill: {variogram_params.get('sill', 0):<11.4f}
      - Nugget: {variogram_params.get('nugget', 0):<30.4f} - Nu: {variogram_params.get('nu', 'N/A'):<13.4f}
    {'='*78}
    CROSS-VALIDATION SCORES
      - S1 (bias): {cv_scores.get('s1', 0):<29.4f} - KGE: {cv_scores.get('kge', 0):<14.4f}
      - S2 (variance): {cv_scores.get('s2', 0):<25.4f} - Correlation: {cv_scores.get('corr', 0):<6.4f}
      - RMSE: {cv_scores.get('rmse', 0):<32.2f} mm - Rel. Bias: {cv_scores.get('rel_bias', 0):<9.2f} %
    {'='*78}
    RECOMMENDATIONS & FLAGS
    """
    if not diagnostics['recommendations']:
        report += "      No major flags detected.\n"
    for rec in diagnostics['recommendations']:
        report += f"      [warn] {rec}\n"
    report += f"    {'='*78}"
    print(report)
    return report

def conditional_kriging_magnitude_ok(
    day: date, df: pd.DataFrame, normalizer: Any,
    variogram_model_name: str, variogram_model_params: dict,
    grid_x: np.ndarray, grid_y: np.ndarray, binary_map: np.ndarray
) -> np.ndarray:
    result_grid = np.copy(binary_map).astype(float)
    wet_mask = (binary_map == 1)
    sub = df[(df["day"] == day) & (df["Rain"] > 0)]
    if len(sub) < 5:
        print(f"  Day {day}: Only {len(sub)} wet stations - using direct assignment")
        result_grid[wet_mask] = 0
        XX, YY = np.meshgrid(grid_x, grid_y, indexing='xy')
        for _, station in sub.iterrows():
            distances = np.sqrt((XX - station['Longitude'])**2 + 
                              (YY - station['Latitude'])**2)
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            result_grid[min_idx] = station['Rain']
        return result_grid
    x_vals, y_vals = sub["Longitude"].values, sub["Latitude"].values
    z_vals = sub["Rain_transformed"].values
    try:
        variogram_function = None
        variogram_parameters = None
        if variogram_model_name == "custom" and "nu" in variogram_model_params:
            variogram_function = _matern_variogram_function
            variogram_parameters = [
                variogram_model_params["nugget"],
                variogram_model_params["sill"] - variogram_model_params["nugget"],
                variogram_model_params["range"],
                variogram_model_params["nu"]
            ]
        else:
            variogram_parameters = [
                variogram_model_params["nugget"],
                variogram_model_params["range"],
                variogram_model_params["sill"]
            ]
        OK = OrdinaryKriging(
            x_vals, y_vals, z_vals,
            variogram_model=variogram_model_name,
            variogram_parameters=variogram_parameters,
            variogram_function=variogram_function,
            verbose=False, enable_plotting=False, pseudo_inv=True
        )
        z, _ = OK.execute('masked', xpoints=grid_x, ypoints=grid_y, mask=~wet_mask)
        non_masked_indices = ~z.mask
        kriged_non_masked = z.data[non_masked_indices]
        back_transformed = normalizer.inverse_transform(
                    kriged_non_masked.reshape(-1, 1)
                ).flatten()
        back_transformed[back_transformed < 0] = 0
        result_grid[non_masked_indices] = back_transformed
    except Exception as e:
        warnings.warn(f"Ordinary Kriging failed for {day}: {str(e)}")
    return result_grid

def conditional_kriging_magnitude_edk(
    day: date, df: pd.DataFrame, normalizer: Any,
    variogram_model_name: str, variogram_model_params: dict,
    grid_x: np.ndarray, grid_y: np.ndarray, binary_map: np.ndarray,
    dem: xr.DataArray
) -> np.ndarray:
    result_grid = np.copy(binary_map).astype(float)
    wet_mask = (binary_map == 1)
    sub = df[(df["day"] == day) & (df["Rain"] > 0)]
    if len(sub) < 4 or not np.any(wet_mask):
        return conditional_kriging_magnitude_ok(day, df, normalizer, variogram_model_name, variogram_model_params, grid_x, grid_y, binary_map)
    x_vals, y_vals = sub["Longitude"].values, sub["Latitude"].values
    z_vals = sub["Rain_transformed"].values
    elev_vals = sub["Elevation"].values
    if np.std(elev_vals) < 1e-3:
        return conditional_kriging_magnitude_ok(day, df, normalizer, variogram_model_name, variogram_model_params, grid_x, grid_y, binary_map)
    try:
        variogram_function = None
        variogram_parameters = None
        if variogram_model_name == "custom" and "nu" in variogram_model_params:
            variogram_function = _matern_variogram_function
            variogram_parameters = [
                variogram_model_params["nugget"],
                variogram_model_params["sill"] - variogram_model_params["nugget"],
                variogram_model_params["range"],
                variogram_model_params["nu"]
            ]
        else:
            variogram_parameters = [
                variogram_model_params["nugget"],
                variogram_model_params["range"],
                variogram_model_params["sill"]
            ]
        UK = UniversalKriging(
            x_vals, y_vals, z_vals,
            variogram_model=variogram_model_name,
            variogram_parameters=variogram_parameters,
            variogram_function=variogram_function,
            drift_terms=['external_Z'],
            external_drift=elev_vals,
            verbose=False, enable_plotting=False, pseudo_inv=True
        )
        z, _ = UK.execute('masked', xpoints=grid_x, ypoints=grid_y, mask=~wet_mask,
                          external_drift_grid=dem.values)
        kriged_values = z.data
        valid_z = np.isfinite(kriged_values)
        if np.any(valid_z):
            back_transformed = normalizer.inverse_transform(kriged_values[valid_z].reshape(-1, 1)).flatten()
            back_transformed[back_transformed < 0] = 0
            temp_grid = np.full(z.shape, np.nan)
            temp_grid[valid_z] = back_transformed
            result_grid[wet_mask] = temp_grid[wet_mask]
    except Exception as e:
        warnings.warn(f"External Drift Kriging failed for {day}: {str(e)}. Falling back to OK.")
        return conditional_kriging_magnitude_ok(day, df, normalizer, variogram_model_name, variogram_model_params, grid_x, grid_y, binary_map)
    return result_grid

def create_magnitude_dataset(kriged_magnitude: Dict[date, np.ndarray],
                             grid_x: np.ndarray, grid_y: np.ndarray) -> xr.Dataset:
    if not kriged_magnitude:
        warnings.warn("No kriged data was generated. Returning empty dataset.")
        return xr.Dataset()
    sorted_dates = sorted(kriged_magnitude.keys())
    magnitude_stack = np.array([kriged_magnitude[d] for d in sorted_dates])
    if magnitude_stack.ndim != 3:
        raise ValueError("Kriged magnitude stack has incorrect dimensions.")
    ds = xr.Dataset(
        {"rainfall_magnitude": (("time", "y", "x"), magnitude_stack)},
        coords={"time": pd.to_datetime(sorted_dates), "x": grid_x, "y": grid_y}
    )
    ds.x.attrs, ds.y.attrs = {"units": "meters", "crs": "EPSG:32633"}, {"units": "meters", "crs": "EPSG:32633"}
    ds.rainfall_magnitude.attrs = {"units": "mm", "long_name": "Rainfall magnitude"}
    return ds

def run_workflow_matern_enhanced(daily_df, final_group_days, occurrence, dem=None, workflow_type="OK"):
    is_edk = (workflow_type == "EDK") and (dem is not None)
    if is_edk:
        print("\n" + "="*70)
        print(f"       STARTING ENHANCED {workflow_type} WORKFLOW (ADAPTIVE)")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print(f"       STARTING ENHANCED {workflow_type} WORKFLOW (ADAPTIVE)")
        print("="*70 + "\n")
    grid_x, grid_y = occurrence.x.values, occurrence.y.values
    kriged_magnitude_all = {}
    save_path = f'outputs/phase2_results/{workflow_type}_enhanced7/'
    os.makedirs(save_path, exist_ok=True)
    DEG_TO_M = 111000.0
    maxlag_deg = 2.0
    bins1 = np.arange(0, 0.15 * DEG_TO_M, 2500)
    bins2 = np.arange(0.15 * DEG_TO_M, 0.4 * DEG_TO_M, 5000)
    bins3 = np.arange(0.4 * DEG_TO_M, 0.8 * DEG_TO_M, 8000)
    bins4 = np.arange(0.8 * DEG_TO_M, 1.3 * DEG_TO_M, 10000)
    bins5 = np.arange(1.3 * DEG_TO_M, maxlag_deg * DEG_TO_M + 1, 15000)
    bin_edges = np.unique(np.concatenate((bins1, bins2, bins3, bins4, bins5)))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for category, days in final_group_days.items():
        print(f"\n{'-'*70}")
        print(f"Processing category: {category} ({len(days)} days)")
        print(f"{'-'*70}")
        if not days:
            print(f"  Empty category {category} - skipping")
            continue
        kriged_magnitude_category = {}
        progress = load_progress(workflow_type, category)
        completed_days = progress.get("completed_days", [])
        remaining_days = [d for d in days if d not in completed_days]
        if not remaining_days and completed_days:
            print(f"  Category {category} already completed. Loading results...")
            try:
                ds_cat = xr.open_dataset(os.path.join(save_path, f"magnitude_{category}.nc"))
                for t in ds_cat.time.values:
                    day = pd.to_datetime(t).date()
                    kriged_magnitude_all[day] = ds_cat['rainfall_magnitude'].sel(time=t).values
                print(f"  Loaded {len(ds_cat.time)} previously kriged days.")
                continue
            except FileNotFoundError:
                print(f"  NetCDF file not found for {category}. Re-running...")
                remaining_days = days
                completed_days = []
            except Exception as e:
                print(f"  Error loading {category} NetCDF: {e}. Re-running...")
                remaining_days = days
                completed_days = []
        if not remaining_days and not completed_days:
            print(f"  No days found for category {category}. Skipping.")
            continue
        elif not remaining_days and completed_days:
            continue
        print(f"  Resume status: {len(completed_days)} days done, {len(remaining_days)} days remaining")
        save_progress(workflow_type, category, completed_days)
        cat_df_raw = daily_df[(daily_df['day'].isin(days)) & (daily_df['Rain'] > 0)].copy()
        if cat_df_raw.empty:
            print(f"  No data found for category {category} - skipping")
            continue
        diagnostics = diagnose_category_quality(cat_df_raw, days, category)
        special_action = handle_problematic_categories(category, diagnostics)
        if special_action == "MERGE_WITH_MODERATE":
            log_skip_reason(None, category, "Insufficient data", f"Avg stations: {diagnostics['avg_stations']:.2f}")
            continue
        if diagnostics['value_statistics']['count'] == 0:
            log_skip_reason(None, category, "No wet data points")
            continue
        print("  Running outlier detection...")
        cat_df = detect_and_handle_outliers(cat_df_raw, diagnostics, days)
        diagnostics_clean = diagnose_category_quality(cat_df, days, category)
        print("  Finding adaptive transformation...")
        cat_rain = cat_df[cat_df['Rain'] > 0]['Rain'].values
        best_transformer, trans_name = adaptive_transformation(cat_rain, diagnostics_clean)
        if best_transformer is None:
            warnings.warn(f"All transformations failed for {category}. Skipping.")
            continue
        best_transformer.fit(cat_rain.reshape(-1, 1))
        print("  Pre-computing transformed values...")
        cat_df_transformed = cat_df.copy()
        positive_mask = cat_df_transformed['Rain']  > 0
        cat_df_transformed.loc[positive_mask, 'Rain_transformed'] = best_transformer.transform(
            cat_df_transformed.loc[positive_mask, 'Rain'].values.reshape(-1, 1)
        ).flatten()
        cat_df_transformed.loc[~positive_mask, 'Rain_transformed'] = 0
        try:
            print("  Calculating empirical variogram...")
            sum_gamma, sum_counts = np.zeros_like(bin_centers), np.zeros_like(bin_centers)
            pbar_vario = tqdm(days, desc="  Empirical Variogram", leave=False)
            for day in pbar_vario:
                sub = cat_df_transformed[(cat_df_transformed['day'] == day) & (cat_df_transformed['Rain']  >0) ]
                if len(sub) < 2:
                    continue
                x, y = sub['Longitude'].values, sub['Latitude'].values
                vals = sub['Rain_transformed'].values
                try:
                    _, gamma_d, counts_d = gs.vario_estimate(
                        (x, y), vals, bin_edges=bin_edges, estimator="cressie", return_counts=True
                    )
                    sum_gamma += np.nan_to_num(gamma_d) * counts_d
                    sum_counts += counts_d
                except Exception:
                    continue
            valid = sum_counts > 0
            if not valid.any():
                warnings.warn(f"Class {category}: no variogram pairs; skipping.")
                continue
            gamma_mean = np.divide(sum_gamma, sum_counts, where=valid, out=np.zeros_like(sum_gamma))
            print("  Fitting adaptive Matern model...")
            gs_mod = gs.Matern(dim=2)
            bounds = get_adaptive_variogram_bounds(category, diagnostics_clean, bin_centers[valid], gamma_mean[valid])
            if special_action == "USE_ROBUST_VARIOGRAM":
                print("  Using robust 'cressie' estimator weights for fit.")
                fit_weights = "cressie"
            else:
                fit_weights = sum_counts[valid]
            gs_mod.set_arg_bounds(**bounds)
            gs_mod.fit_variogram(bin_centers[valid], gamma_mean[valid], weights=fit_weights, loss="linear", max_eval=100000)
            model_name, model_params = _translate_gstools_to_pykrige(gs_mod)
            print("  Running adaptive cross-validation...")
            cv_scores = adaptive_cross_validation(
                cat_df_transformed, days, model_name, model_params,
                best_transformer, diagnostics_clean, is_edk
            )
            report = enhanced_model_reporting(category, diagnostics_clean, cv_scores, model_params, trans_name)
            save_enhanced_report(report, category, workflow_type, save_path)
            print(f"  Kriging {len(remaining_days)} remaining days...")
            pbar_krig = tqdm(remaining_days, desc=f"  Kriging ({category})", unit="day")
            for i, day in enumerate(pbar_krig):
                try:
                    binary_map = occurrence['binary_map'].sel(time=pd.to_datetime(day)).values
                    day_data = cat_df_transformed[cat_df_transformed['day'] == day]
                    if is_edk:
                        rainfall_grid = conditional_kriging_magnitude_edk(
                            day, day_data, best_transformer, model_name, model_params,
                            grid_x, grid_y, binary_map, dem
                        )
                    else:
                        rainfall_grid = conditional_kriging_magnitude_ok(
                            day, day_data, best_transformer, model_name, model_params,
                            grid_x, grid_y, binary_map
                        )
                    kriged_magnitude_category[day] = rainfall_grid
                    completed_days.append(day)
                    save_progress(workflow_type, category, completed_days)
                    if (i + 1) % 5 == 0:
                        if kriged_magnitude_category:
                            try:
                                ds_temp = create_magnitude_dataset(kriged_magnitude_category, grid_x, grid_y)
                                temp_file = os.path.join(save_path, f"magnitude_{category}_TEMP.nc")
                                ds_temp.to_netcdf(temp_file)
                                print(f"  Saved temporary results ({len(kriged_magnitude_category)} days)")
                            except Exception as e:
                                print(f"  Could not save temp file: {e}")
                except Exception as e:
                    print(f"  Error on {day}: {e}")
                    save_progress(workflow_type, category, completed_days)
                    continue
            if kriged_magnitude_category:
                ds_cat = create_magnitude_dataset(kriged_magnitude_category, grid_x, grid_y)
                ds_cat.to_netcdf(os.path.join(save_path, f"magnitude_{category}th0loocv.nc"))
                print(f"  Saved final results: magnitude_{category}.nc")
                kriged_magnitude_all.update(kriged_magnitude_category)
                cleanup_progress(workflow_type, category)
                temp_file = os.path.join(save_path, f"magnitude_{category}_TEMP.nc")
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"  Removed temporary file")
                    except:
                        pass
        except Exception as e:
            warnings.warn(f"FATAL ERROR for category {category}: {e}")
            print(f"  Progress saved. Resume will continue from here.")
        gc.collect()
    print(f"\n--- Saving final combined {workflow_type} results ---")
    if kriged_magnitude_all:
        magnitude_ds = create_magnitude_dataset(kriged_magnitude_all, grid_x, grid_y)
        final_filename = f"rainfall_magnitude_{workflow_type}_30.11.2025.nc"
        magnitude_ds.to_netcdf(final_filename)
        print(f"Final combined dataset saved to {final_filename}")
    else:
        print("No kriged data was generated in this run.")

if __name__ == '__main__':
    print("Testing progress system...")
    test_date = date(2023, 1, 1)
    save_progress("TEST", "TEST_CATEGORY", [test_date])
    progress_data = load_progress("TEST", "TEST_CATEGORY")
    print(f"Progress test: {len(progress_data['completed_days'])} days loaded")
    cleanup_progress("TEST", "TEST_CATEGORY")
    print("Progress system working correctly\n")
    print("Checking for interrupted runs...")
    recover_interrupted_run("OK")
    recover_interrupted_run("EDK")
    print()
    try:
        daily_df_raw = pd.read_pickle("data/input/daily_df_m.pkl")
        with open('outputs/phase2_calibration/final_group_daysII.pkl ', 'rb') as f:
            final_group_days = pickle.load(f)
        occurrence_ds = xr.open_dataset("outputs/phase2_calibration/rainfall_occurrence_28.11.2025.nc")
        dem_ds = xr.open_dataset("data/input/dem.nc")['dem']
        metadata_df = pd.read_excel("data/input/metadata.xlsx")
        print("All data loaded.")
    except FileNotFoundError as e:
        print(f"FATAL: Could not find data file: {e}. Exiting.")
    except Exception as e:
        print(f"FATAL: Error loading data: {e}. Exiting.")
    print("Loading and merging station elevation data...")
    daily_df_with_elev = pd.merge(daily_df_raw, metadata_df[['ID', 'Elevation']], on='ID', how='left')
    missing_ids = daily_df_with_elev[daily_df_with_elev['Elevation'].isnull()]['ID'].unique()
    if len(missing_ids) > 0:
        warnings.warn(f"Warning: {len(missing_ids)} stations miss elevation data. They will be excluded from EDK.")
        daily_df_for_edk = daily_df_with_elev.dropna(subset=['Elevation']).copy()
    else:
        daily_df_for_edk = daily_df_with_elev.copy()
    print("Station elevation data successfully merged.")
    if 'daily_df_raw' in locals() and 'final_group_days' in locals() and 'occurrence_ds' in locals():
        run_workflow_matern_enhanced(
            daily_df_raw, 
            final_group_days, 
            occurrence_ds, 
            dem=None, 
            workflow_type="OK"
        )
    else:
        print("Skipping OK workflow, data not loaded.")
    if 'daily_df_for_edk' in locals() and 'final_group_days' in locals() and 'occurrence_ds' in locals() and 'dem_ds' in locals():
        run_workflow_matern_enhanced(
            daily_df_for_edk, 
            final_group_days, 
            occurrence_ds, 
            dem=dem_ds, 
            workflow_type="EDK"
        )
    else:
        print("Skipping EDK workflow, data not loaded.")
    print("\n" + "="*50)
    print("    ALL ENHANCED WORKFLOWS COMPLETED")
    print("="*50 + "\n")
