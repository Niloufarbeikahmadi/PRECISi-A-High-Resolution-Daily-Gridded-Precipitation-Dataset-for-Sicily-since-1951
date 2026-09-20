### code for: Phase I, calibration of rain Occurrence model ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gstools as gs
from pykrige.ok import OrdinaryKriging
from typing import List, Tuple, Dict
from tqdm import tqdm
import datetime
import seaborn as sns
import geopandas as gpd
import pickle
import logging
from scipy.spatial.distance import cdist
import xarray as xr
import warnings
logger = logging.getLogger(__name__)
def binary_indicator(daily_df: pd.DataFrame, trace_threshold: float = 0.2) -> pd.DataFrame:
    daily_df['indicator'] = (daily_df['Rain'] > trace_threshold).astype(int)
    return daily_df
def compute_cutoff_threshold(prob_map: np.ndarray) -> float:
    p = prob_map.flatten()
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return 0.5
    m = np.mean(p)
    p_sorted = np.sort(p)
    N = len(p_sorted)
    if m <= 0:
        return 1.0
    if m >= 1:
        return 0.0
    quantile_index = int(np.floor((1.0 - m) * N))
    quantile_index = max(0, min(quantile_index, N - 1))
    cutoff = p_sorted[quantile_index]
    return cutoff
def calculate_class_cutoffs(results: Dict, class_dict: Dict[str, List]) -> Dict[str, float]:
    class_probs = {label: [] for label in class_dict}
    for day, data in results.items():
        prob_map = data['prob_map']
        day_ts = pd.Timestamp(day)
        for class_label, dates in class_dict.items():
            date_list = [pd.Timestamp(d) for d in dates]
            if day_ts in date_list:
                class_probs[class_label].append(prob_map)
    class_thresholds = {}
    for class_label, prob_maps in class_probs.items():
        if prob_maps:
            all_probs = np.concatenate([pm[~np.isnan(pm)] for pm in prob_maps])
            if len(all_probs) > 0:
                class_thresholds[class_label] = compute_cutoff_threshold(all_probs)
            else:
                class_thresholds[class_label] = 0.5
                logger.warning(f"No valid probability values for class {class_label}, using default threshold 0.5")
    return class_thresholds
def create_binary_maps(occurrence_results, class_thresholds, class_dict):
    for day_key, data in occurrence_results.items():
        day_ts = pd.Timestamp(day_key)
        found_class = None
        for class_label, days in class_dict.items():
            day_list = [pd.Timestamp(d) for d in days]
            if day_ts in day_list:
                found_class = class_label
                break
        if found_class is None:
            logger.warning(f"No class found for day {day_key}")
            continue
        threshold = class_thresholds.get(found_class)
        if threshold is None:
            logger.warning(f"No threshold for class {found_class}")
            continue
        prob_map = data.get('prob_map')
        if prob_map is None:
            logger.warning(f"No probability map for day {day_key}")
            continue
        binary_map = np.full_like(prob_map, np.nan, dtype=float)
        valid_mask = ~np.isnan(prob_map)
        binary_map[valid_mask] = (prob_map[valid_mask] > threshold).astype(int)
        occurrence_results[day_key]['binary_map'] = binary_map
        occurrence_results[day_key]['cutoff'] = threshold
    return occurrence_results
def classify_days(daily_df: pd.DataFrame, thresholds: List[Tuple[float, float]], 
                  trace_threshold: float = 0.2) -> Dict[str, List]:
    day_groups = daily_df.groupby('day')
    dry_freq = day_groups.apply(
        lambda g: (g['Rain'] <= trace_threshold).sum() / len(g) * 100, 
        include_groups=False
    )
    class_dict = {}
    for th in thresholds:
        label = f"F{int(th[0])}-{int(th[1])}"
        class_days = dry_freq[(dry_freq >= th[0]) & (dry_freq <= th[1])].index.tolist()
        class_dict[label] = class_days
    all_classified_days = set(sum(class_dict.values(), []))
    unclassified_days = set(dry_freq.index) - all_classified_days
    if unclassified_days:
        logger.warning(f"Unclassified days: {sorted(list(unclassified_days))}")
    return class_dict
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
def _matern_variogram_function(params, dist):
    nugget, psill, range_val, nu = params
    model = gs.Matern(dim=2, var=psill, len_scale=range_val, nugget=nugget, nu=nu)
    return model.variogram(dist)
def compute_empirical_variogram_class(daily_df: pd.DataFrame, days: List, 
                                     maxlag: float = 200000, n_lags: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    DEG_TO_M = 111000.0
    maxlag_deg = maxlag / DEG_TO_M
    bins1 = np.arange(0, 0.15 * DEG_TO_M, 2500)       
    bins2 = np.arange(0.15 * DEG_TO_M, 0.4 * DEG_TO_M, 5000)  
    bins3 = np.arange(0.4 * DEG_TO_M, 0.8 * DEG_TO_M, 8000)   
    bins4 = np.arange(0.8 * DEG_TO_M, 1.3 * DEG_TO_M, 10000)  
    bins5 = np.arange(1.3 * DEG_TO_M, maxlag_deg * DEG_TO_M + 1, 15000)   
    bin_edges = np.unique(np.concatenate((bins1, bins2, bins3, bins4, bins5)))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    sum_gamma = np.zeros_like(bin_centers)
    sum_counts = np.zeros_like(bin_centers)
    for day in tqdm(days, desc="Computing empirical variogram"):
        sub = daily_df[daily_df['day'] == day]
        if len(sub) < 2:
            continue
        coords = sub[['Longitude', 'Latitude']].values
        vals = sub['indicator'].values.astype(float)
        try:
            bin_center, gamma, counts = gs.vario_estimate(
                (coords[:, 0], coords[:, 1]), vals, 
                bin_edges=bin_edges, estimator="matheron", 
                return_counts=True
            )
            sum_gamma += np.nan_to_num(gamma) * counts
            sum_counts += counts
        except Exception as e:
            logger.warning(f"Variogram estimation failed for day {day}: {e}")
            continue
    valid = sum_counts > 0
    if not valid.any():
        return None, None, None
    gamma_mean = np.divide(sum_gamma, sum_counts, where=valid, 
                          out=np.zeros_like(sum_gamma))
    return bin_centers[valid], gamma_mean[valid], sum_counts[valid]
def fit_variogram_models(bin_centers: np.ndarray, gamma: np.ndarray, 
                        class_label: str) -> Dict[str, Tuple]:
    models = {
        'Spherical': gs.Spherical,
        'Exponential': gs.Exponential, 
        'Gaussian': gs.Gaussian,
        'Matern': gs.Matern
    }
    fitted_models = {}
    for model_name, model_class in models.items():
        try:
            if model_name == 'Matern':
                model = model_class(dim=2)
                model.set_arg_bounds(nu=[0.2, 5.0])
            else:
                model = model_class(dim=2)
            model.fit_variogram(bin_centers, gamma, 
                              weights=np.ones_like(bin_centers), 
                              loss="linear")
            fitted_models[model_name] = model
            logger.info(f"  {model_name} fitted for {class_label}")
        except Exception as e:
            logger.warning(f"  {model_name} fitting failed for {class_label}: {e}")
    return fitted_models
def daily_loocv(daily_df: pd.DataFrame, days: List, model_name: str, 
                model_params: dict, class_label: str) -> Dict[str, float]:
    all_obs, all_pred, all_variances = [], [], []
    for day in tqdm(days, desc=f"LOOCV for {class_label}"):
        sub = daily_df[daily_df['day'] == day]
        if len(sub) < 3:
            continue
        coords = sub[['Longitude', 'Latitude']].values
        indicators = sub['indicator'].values.astype(float)
        predictions = np.full_like(indicators, np.nan)
        variances = np.full_like(indicators, np.nan)
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
        for i in range(len(coords)):
            try:
                mask = np.ones(len(coords), dtype=bool)
                mask[i] = False
                x_train, y_train = coords[mask, 0], coords[mask, 1]
                z_train = indicators[mask]
                if np.std(z_train) < 1e-10:
                    continue
                OK = OrdinaryKriging(
                    x_train, y_train, z_train,
                    variogram_model=model_name,
                    variogram_parameters=variogram_parameters,
                    variogram_function=variogram_function,
                    verbose=False, enable_plotting=False,
                    pseudo_inv=True
                )
                pred, var = OK.execute('points', [coords[i, 0]], [coords[i, 1]])
                predictions[i] = pred[0]
                if var[0] is not None and var[0] > 1e-10:
                    variances[i] = max(var[0], 1e-10)
                else:
                    variances[i] = np.nan
            except Exception as e:
                continue
        valid_mask = np.isfinite(predictions) & np.isfinite(variances) & (variances > 1e-10)
        if np.any(valid_mask):
            all_obs.extend(indicators[valid_mask])
            all_pred.extend(predictions[valid_mask])
            all_variances.extend(variances[valid_mask])
    if not all_obs or len(all_obs) < 10:
        logger.warning(f"Insufficient valid LOOCV points for {class_label}: {len(all_obs)}")
        return {}
    obs = np.array(all_obs)
    pred = np.array(all_pred)
    var = np.array(all_variances)
    residuals = obs - pred
    with np.errstate(invalid='ignore', divide='ignore'):
        standard_errors = np.sqrt(var)
        valid_se = standard_errors > 1e-10
        standardized_residuals = np.full_like(residuals, np.nan)
        standardized_residuals[valid_se] = residuals[valid_se] / standard_errors[valid_se]
    valid_std_resid = np.isfinite(standardized_residuals)
    if not np.any(valid_std_resid):
        logger.warning(f"No valid standardized residuals for {class_label}")
        return {}
    standardized_residuals = standardized_residuals[valid_std_resid]
    s1 = np.mean(standardized_residuals)
    s2 = np.mean(standardized_residuals ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    brier_score = np.mean((pred - obs) ** 2)
    return {
        's1': float(s1),
        's2': float(s2), 
        'rmse': float(rmse),
        'brier_score': float(brier_score),
        'n_points': len(all_obs)
    }
def select_best_variogram(daily_df: pd.DataFrame, class_dict: Dict[str, List], 
                         trace_threshold: float = 0.2) -> Tuple[Dict[str, Tuple], Dict]:
    best_models = {}
    class_info = {}
    for class_label, days in class_dict.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing class: {class_label}")
        logger.info(f"{'='*50}")
        class_info[class_label] = {
            'days_count': len(days),
            'empirical_variogram': {},
            'fitted_models': {},
            'best_model': None,
            'cutoff_threshold': None
        }
        bin_centers, gamma, counts = compute_empirical_variogram_class(daily_df, days)
        if bin_centers is None:
            logger.warning(f"No valid variogram for {class_label}")
            continue
        class_info[class_label]['empirical_variogram'] = {
            'bin_centers': bin_centers,
            'gamma': gamma,
            'counts': counts
        }
        fitted_models = fit_variogram_models(bin_centers, gamma, class_label)
        if not fitted_models:
            logger.warning(f"No models fitted for {class_label}")
            continue
        model_scores = {}
        best_model_name = None
        best_model_params = None
        best_pykrige_model = None
        for model_name, gs_model in fitted_models.items():
            logger.info(f"  Evaluating {model_name} with LOOCV...")
            pykrige_model, model_params = _translate_gstools_to_pykrige(gs_model)
            class_info[class_label]['fitted_models'][model_name] = {
                'gs_model': gs_model,
                'pykrige_model': pykrige_model,
                'model_params': model_params
            }
            cv_results = daily_loocv(daily_df, days, pykrige_model, 
                                   model_params, class_label)
            if cv_results and np.isfinite(cv_results.get('s2', np.nan)):
                score = abs(cv_results['s2'] - 1)
                model_scores[model_name] = score
                class_info[class_label]['fitted_models'][model_name]['cv_results'] = cv_results
                logger.info(f"    {model_name}: S1={cv_results['s1']:.3f}, "
                           f"S2={cv_results['s2']:.3f}, Score={score:.3f}, "
                           f"n_points={cv_results.get('n_points', 0)}")
                if best_model_name is None or score < model_scores[best_model_name]:
                    best_model_name = model_name
                    best_model_params = model_params
                    best_pykrige_model = pykrige_model
            else:
                logger.warning(f"    LOOCV failed for {model_name}")
                class_info[class_label]['fitted_models'][model_name]['cv_results'] = {}
        if best_pykrige_model and best_model_params:
            best_models[class_label] = (best_pykrige_model, best_model_params)
            class_info[class_label]['best_model'] = {
                'model_name': best_model_name,
                'pykrige_model': best_pykrige_model,
                'model_params': best_model_params,
                'selection_score': model_scores[best_model_name] if best_model_name in model_scores else None
            }
            logger.info(f"  SELECTED: {best_model_name} (S2 closest to 1)")
        else:
            logger.warning(f"  No suitable model found for {class_label}")
    return best_models, class_info
def krige_occurrence_probability(daily_df: pd.DataFrame, class_dict: Dict[str, List],
                                best_models: Dict[str, Tuple], dem_path: str = None) -> Dict:
    if dem_path is None:
        dem_path = "outputs/phase2_calibration/dem.nc"
    try:
        dem = xr.open_dataset(dem_path)['dem']
        grid_lons = dem.x.values
        grid_lats = dem.y.values
        land_mask = ~np.isnan(dem.values)
        logger.info(f"Grid dimensions from DEM: {len(grid_lons)} x {len(grid_lats)}")
        logger.info(f"X range: {grid_lons.min():.0f} to {grid_lons.max():.0f}")
        logger.info(f"Y range: {grid_lats.min():.0f} to {grid_lats.max():.0f}")
        logger.info(f"Land mask: {np.sum(land_mask)} land cells, {np.sum(~land_mask)} sea cells")
    except Exception as e:
        logger.error(f"Failed to load DEM grid: {e}")
        grid_lats = np.arange(4044000, 4263000, 2000)
        grid_lons = np.arange(231300, 588500, 2000)
        land_mask = np.ones((len(grid_lats), len(grid_lons)), dtype=bool)
        logger.warning("Using fallback grid coordinates")
    results = {}
    day_to_class = {}
    for class_label, days in class_dict.items():
        for day in days:
            day_to_class[pd.Timestamp(day)] = class_label
    processed_days = 0
    skipped_days = 0
    for day in tqdm(daily_df['day'].unique(), desc="Kriging occurrence probability"):
        sub = daily_df[daily_df['day'] == day].drop_duplicates(
            subset=['Latitude', 'Longitude']
        )
        if len(sub) < 3 or sub['indicator'].nunique() <= 1:
            if len(sub) > 0:
                indicator_value = sub['indicator'].unique()[0]
                constant_map = np.full((len(grid_lats), len(grid_lons)), np.nan)
                constant_map[land_mask] = indicator_value
                results[day] = {
                    'grid_lons': grid_lons,
                    'grid_lats': grid_lats,
                    'prob_map': constant_map,
                }
                processed_days += 1
            else:
                nan_map = np.full((len(grid_lats), len(grid_lons)), np.nan)
                results[day] = {
                    'grid_lons': grid_lons,
                    'grid_lats': grid_lats,
                    'prob_map': nan_map,
                }
                processed_days += 1
            continue
        class_label = day_to_class.get(pd.Timestamp(day))
        if not class_label or class_label not in best_models:
            nan_map = np.full((len(grid_lats), len(grid_lons)), np.nan)
            results[day] = {
                'grid_lons': grid_lons,
                'grid_lats': grid_lats,
                'prob_map': nan_map,
            }
            processed_days += 1
            continue
        model_name, model_params = best_models[class_label]
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
            OK = OrdinaryKriging(
                x=sub['Longitude'].values,
                y=sub['Latitude'].values,
                z=sub['indicator'].values.astype(float),
                variogram_model=model_name,
                variogram_parameters=variogram_parameters,
                variogram_function=variogram_function,
                exact_values=True,
            )
            z, ss = OK.execute('grid', grid_lons, grid_lats)
            z[~land_mask] = np.nan
            results[day] = {
                'grid_lons': grid_lons,
                'grid_lats': grid_lats,
                'prob_map': z,
            }
            processed_days += 1
        except Exception as e:
            logger.warning(f"Skipping day {day} due to error: {e}")
            nan_map = np.full((len(grid_lats), len(grid_lons)), np.nan)
            results[day] = {
                'grid_lons': grid_lons,
                'grid_lats': grid_lats,
                'prob_map': nan_map,
            }
            processed_days += 1
    logger.info(f"Kriging completed: {processed_days} days processed, {skipped_days} days skipped")
    return results
def create_occurrence_dataset(occurrence_results: Dict, grid_x: np.ndarray, grid_y: np.ndarray, 
                             dem_path: str = None) -> xr.Dataset:
    if not occurrence_results:
        warnings.warn("No occurrence data was generated. Returning empty dataset.")
        return xr.Dataset()
    sorted_dates = sorted(occurrence_results.keys())
    prob_stack = []
    binary_stack = []
    for date in sorted_dates:
        data = occurrence_results[date]
        prob_stack.append(data['prob_map'])
        if 'binary_map' in data:
            binary_stack.append(data['binary_map'])
        else:
            nan_map = np.full_like(data['prob_map'], np.nan)
            binary_stack.append(nan_map)
    prob_stack = np.array(prob_stack)
    binary_stack = np.array(binary_stack)
    if prob_stack.ndim != 3 or binary_stack.ndim != 3:
        raise ValueError("Occurrence stacks have incorrect dimensions.")
    if prob_stack.shape != binary_stack.shape:
        raise ValueError("Probability and binary maps have different shapes.")
    ds = xr.Dataset(
        {
            "prob_map": (("time", "y", "x"), prob_stack),
            "binary_map": (("time", "y", "x"), binary_stack),
        },
        coords={
            "time": pd.to_datetime(sorted_dates), 
            "x": grid_x, 
            "y": grid_y
        }
    )
    if dem_path is not None:
        try:
            dem = xr.open_dataset(dem_path)['dem']
            land_mask = ~np.isnan(dem.values)
            ds["land_mask"] = (("y", "x"), land_mask)
            logger.info("Added land mask from DEM")
        except Exception as e:
            logger.warning(f"Could not load land mask from DEM: {e}")
    ds.x.attrs = {"units": "meters", "crs": "EPSG:32633"}
    ds.y.attrs = {"units": "meters", "crs": "EPSG:32633"}
    ds.time.attrs = {"long_name": "Time"}
    ds.prob_map.attrs = {
        "units": "probability", 
        "long_name": "Probability of rainfall occurrence",
        "description": "Probability map from indicator kriging"
    }
    ds.binary_map.attrs = {
        "units": "binary", 
        "long_name": "Binary rainfall occurrence",
        "description": "Binary rain (1) / no rain (0) map"
    }
    if "land_mask" in ds:
        ds.land_mask.attrs = {
            "units": "binary",
            "long_name": "Land mask",
            "description": "True for land, False for sea"
        }
    return ds
def main():
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        daily_df = pd.read_pickle("data/input/daily_df_m.pkl")
        trace_threshold = 0.2
        thresholds = [(0, 25), (25, 75), (75, 100)]
        class_dict = classify_days(daily_df, thresholds, trace_threshold)
        daily_df = binary_indicator(daily_df, trace_threshold)
        logger.info("Starting variogram model selection with LOOCV...")
        best_models, class_info = select_best_variogram(daily_df, class_dict, trace_threshold)
        logger.info("Starting indicator kriging with DEM grid alignment...")
        occurrence_results = krige_occurrence_probability(
            daily_df, class_dict, best_models, 
            dem_path="outputs/phase2_calibration/dem.nc"
        )
        logger.info("Calculating class thresholds...")
        class_thresholds = calculate_class_cutoffs(occurrence_results, class_dict)
        for class_label, threshold in class_thresholds.items():
            if class_label in class_info:
                class_info[class_label]['cutoff_threshold'] = threshold
        logger.info("Creating binary maps...")
        results_with_binary = create_binary_maps(occurrence_results, class_thresholds, class_dict)
        if results_with_binary:
            first_day = list(results_with_binary.keys())[0]
            grid_x = results_with_binary[first_day]['grid_lons']
            grid_y = results_with_binary[first_day]['grid_lats']
            logger.info("Creating final xarray dataset...")
            occurrence_ds = create_occurrence_dataset(
                results_with_binary, 
                grid_x, 
                grid_y,
                dem_path="outputs/phase2_calibration/dem.nc"
            )
            from datetime import datetime
            current_date = datetime.now().strftime("%d.%m.%Y")
            output_file = f"rainfall_occurrence_{current_date}.nc"
            occurrence_ds.to_netcdf(output_file)
            logger.info(f"Saved occurrence dataset to {output_file}")
            logger.info("Final dataset structure:")
            logger.info(f"Dimensions: {dict(occurrence_ds.dims)}")
            logger.info(f"Coordinates:")
            for coord in occurrence_ds.coords:
                logger.info(f"  {coord}: {occurrence_ds[coord].values.shape} {occurrence_ds[coord].values.dtype}")
            logger.info(f"Data variables:")
            for var in occurrence_ds.data_vars:
                logger.info(f"  {var}: {occurrence_ds[var].values.shape} {occurrence_ds[var].values.dtype}")
        logger.info("Saving additional results...")
        with open("class_dict_20.11.2025.pkl", 'wb') as f:
            pickle.dump(class_dict, f)
        with open("occurrence_results_20.11.2025.pkl", 'wb') as f:
            pickle.dump(results_with_binary, f)
        with open("class_info_20.11.2025.pkl", 'wb') as f:
            pickle.dump(class_info, f)
        logger.info("Phase I completed successfully!")
    except Exception as e:
        logger.error(f"Phase I failed: {e}")
        raise
if __name__ == "__main__":
    main()
