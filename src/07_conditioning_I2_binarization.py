### code for ECDF and I2 binarization (calibration + application) ###

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pickle
import random
from pathlib import Path
from scipy import interpolate
import warnings

random.seed(42)

base_path = Path(".")
occ_path = Path("outputs/phase2_calibration/rainfall_occurrence_28.11.2025.nc")
class_dict_path = Path("data/input/class_dict_27.11.2025.pkl")
daily_df_path = Path("data/input/daily_df_m.pkl")
output_dir = Path("outputs/phase2_calibration")
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading occurrence dataset...")
ds = xr.open_dataset(occ_path)
original_binary = ds.binary_map.values.copy()

print("Loading class dictionary...")
with open(class_dict_path, 'rb') as f:
    class_dict = pickle.load(f)

print("Loading daily station data...")
daily_df = pd.read_pickle(daily_df_path)
daily_df['day'] = pd.to_datetime(daily_df['day'])

daily_df['dry'] = (daily_df['Rain'] <= 0).astype(int)
daily_stats = daily_df.groupby('day').agg(
    total_stations=('ID', 'count'),
    dry_count=('dry', 'sum')
)
daily_stats['Fdry'] = daily_stats['dry_count'] / daily_stats['total_stations']
daily_Fdry = daily_stats['Fdry']

land_mask = ds.land_mask.values
land_indices = np.where(land_mask.ravel())[0]

time_dates = pd.to_datetime(ds.time.values).date
prob_land_sorted = {}

print("Extracting probability maps on land...")
for t_idx, date in enumerate(time_dates):
    prob_2d = ds.prob_map.isel(time=t_idx).values
    prob_flat = prob_2d.ravel()
    prob_land = prob_flat[land_indices]
    prob_land_sorted[date] = np.sort(prob_land)

date_to_class = {}
for class_name, dates in class_dict.items():
    for d in dates:
        if isinstance(d, pd.Timestamp):
            d = d.date()
        date_to_class[d] = class_name

class_dates = {cls: [] for cls in class_dict.keys()}
for date, cls in date_to_class.items():
    class_dates[cls].append(date)

threshold_grid = np.linspace(0, 1, 1000)
allocation_funcs = {}

for cls_name, dates in class_dates.items():
    print(f"Computing allocation function for class {cls_name}...")
    cum_list = []
    valid_dates = [d for d in dates if d in prob_land_sorted]
    if len(valid_dates) == 0:
        print(f"  Warning: no probability maps found for class {cls_name}")
        continue
    for date in valid_dates:
        sorted_probs = prob_land_sorted[date]
        n_pixels = len(sorted_probs)
        cum = np.searchsorted(sorted_probs, threshold_grid, side='right') / n_pixels
        cum_list.append(cum)
    cum_array = np.vstack(cum_list)
    median_cum = np.median(cum_array, axis=0)
    allocation_funcs[cls_name] = {
        'thresholds': threshold_grid.copy(),
        'cum_median': median_cum,
        'n_days': len(valid_dates)
    }

alloc_path = output_dir / "allocation_functions_calib.pkl"
with open(alloc_path, 'wb') as f:
    pickle.dump(allocation_funcs, f)
print(f"Allocation functions saved to {alloc_path}")

fig, ax = plt.subplots(figsize=(8,6))
for cls_name, alloc in allocation_funcs.items():
    ax.plot(alloc['thresholds'], alloc['cum_median'], label=f"{cls_name} (n={alloc['n_days']})")
ax.set_xlabel("Probability threshold")
ax.set_ylabel("Cumulative probability (median ECDF)")
ax.set_title("Allocation functions (median ECDF per class)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
stats_text = "\n".join([f"{cls}: {alloc['n_days']} days" for cls, alloc in allocation_funcs.items()])
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plot_path = output_dir / "allocation_functions.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Allocation functions plot saved to {plot_path}")

Fdry_dict = {d.date(): f for d, f in daily_Fdry.items()}

inv_alloc = {}
for cls_name, alloc in allocation_funcs.items():
    xp = alloc['cum_median']
    fp = alloc['thresholds']
    inv_alloc[cls_name] = lambda target, xp=xp, fp=fp: np.interp(target, xp, fp, left=0.0, right=1.0)

log_entries = []
new_binary = np.zeros_like(ds.binary_map.values, dtype=np.float64)

print("Processing each day to compute dynamic thresholds and binarize...")
for t_idx, date in enumerate(time_dates):
    if date not in date_to_class:
        print(f"  Warning: date {date} not found in class dictionary, skipping")
        continue
    cls_name = date_to_class[date]
    if date not in Fdry_dict:
        print(f"  Warning: no Fdry for date {date}, skipping")
        continue
    Fdry = Fdry_dict[date]
    expected_range = {
        'F0-25': (0.0, 0.25),
        'F25-75': (0.25, 0.75),
        'F75-100': (0.75, 1.0)
    }
    low, high = expected_range[cls_name]
    out_of_range = not (low <= Fdry <= high)
    if out_of_range:
        print(f"  NOTE: Date {date} class {cls_name} has Fdry = {Fdry:.3f} outside expected range [{low}, {high}]")
    target_cum = Fdry
    inv_func = inv_alloc[cls_name]
    threshold = inv_func(target_cum)
    prob_slice = ds.prob_map.isel(time=t_idx).values
    binary_slice = (prob_slice > threshold).astype(np.float64)
    binary_slice[~land_mask] = np.nan
    new_binary[t_idx, :, :] = binary_slice
    log_entries.append({
        'date': date,
        'class': cls_name,
        'Fdry': Fdry,
        'threshold': threshold,
        'out_of_range': out_of_range
    })

log_df = pd.DataFrame(log_entries)
log_path = output_dir / "threshold_log_calib.pkl"
log_df.to_pickle(log_path)
print(f"Log saved to {log_path}")

ds['binary_map'] = (('time', 'y', 'x'), new_binary)
ds.binary_map.attrs['long_name'] = 'Binary rain occurrence (1 = rain, 0 = no rain)'
ds.binary_map.attrs['description'] = 'Generated using dynamic thresholds from allocation functions'

output_nc = output_dir / "occurrence_calib_15.2.2026.nc"
ds.to_netcdf(output_nc)
print(f"Updated dataset saved to {output_nc}")

random.seed(42)
n_random = 3
fig_dir = output_dir / "daily_ecdf_plots"
fig_dir.mkdir(exist_ok=True)

for cls_name, dates in class_dates.items():
    valid_dates = [d for d in dates if d in prob_land_sorted]
    if len(valid_dates) < n_random:
        print(f"Class {cls_name} has only {len(valid_dates)} days, taking all for random plots")
        selected = valid_dates
    else:
        selected = random.sample(valid_dates, n_random)
    for date in selected:
        sorted_probs = prob_land_sorted[date]
        x_ecdf = sorted_probs
        y_ecdf = np.arange(1, len(sorted_probs)+1) / len(sorted_probs)
        alloc = allocation_funcs[cls_name]
        thresh_grid = alloc['thresholds']
        cum_median = alloc['cum_median']
        entry = log_df[log_df['date'] == date].iloc[0]
        Fdry_val = entry['Fdry']
        thresh_val = entry['threshold']
        FW_val = 1 - Fdry_val
        fig, ax = plt.subplots(figsize=(8,6))
        ax.step(x_ecdf, y_ecdf, where='post', label=f'Daily ECDF ({date})', alpha=0.8)
        ax.plot(thresh_grid, cum_median, 'r-', linewidth=2, label=f'{cls_name} median ECDF')
        ax.axvline(thresh_val, color='k', linestyle='--', label=f'Threshold = {thresh_val:.3f}')
        info_text = f"Fdry = {Fdry_val:.3f}\nFW = {FW_val:.3f}"
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_xlabel("Probability of rain")
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"Class {cls_name} - Date {date}")
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = fig_dir / f"{cls_name}_{date}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved ECDF plot for {cls_name} {date}")

print("All tasks completed.")

print("\nGenerating before-after comparison plots for random dates...")

n_compare = 5
if len(log_df) >= n_compare:
    compare_indices = random.sample(range(len(log_df)), n_compare)
    compare_dates = log_df.iloc[compare_indices]['date'].values
    fig, axes = plt.subplots(n_compare, 2, figsize=(12, 4*n_compare))
    if n_compare == 1:
        axes = axes.reshape(1, -1)
    for i, date in enumerate(compare_dates):
        time_idx = np.where(pd.to_datetime(ds.time.values).date == date)[0]
        if len(time_idx) == 0:
            print(f"  Date {date} not found in dataset, skipping")
            continue
        t = time_idx[0]
        orig_slice = original_binary[t, :, :]
        new_slice = new_binary[t, :, :]
        entry = log_df[log_df['date'] == date].iloc[0]
        cls = entry['class']
        Fdry = entry['Fdry']
        thresh = entry['threshold']
        ax = axes[i, 0]
        im0 = ax.imshow(orig_slice, cmap='RdBu', vmin=0, vmax=1, aspect='auto', origin='lower')
        ax.set_title(f"Original binary map\n{date} (Class: {cls})", fontsize=10)
        ax.axis('off')
        ax = axes[i, 1]
        im1 = ax.imshow(new_slice, cmap='RdBu', vmin=0, vmax=1, aspect='auto', origin='lower')
        ax.set_title(f"New binary map (dynamic threshold)\nFdry={Fdry:.3f}, thresh={thresh:.3f}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    comp_plot_path = output_dir / "before_after_comparison.png"
    plt.savefig(comp_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Before-after comparison plot saved to {comp_plot_path}")
else:
    print("Not enough days in log for comparison plots.")

# ------------------------------------------------------------
# Historical application part

base_dir = Path("outputs/reconstruction_IX")
alloc_path = base_dir / "allocation_functions_calib.pkl"
class_dict_path = base_dir / "class_dict_ALL.pkl"
combined_data_path = base_dir / "combined_dataset.pkl"

prob_dir = base_dir / "Phase_I/Occurrence_Probability_Grids"
output_dir = base_dir / "Phase_I/Binary_Occurrence_Grids"
output_dir.mkdir(parents=True, exist_ok=True)

expected_ranges = {
    'F0-25': (0.0, 0.25),
    'F25-75': (0.25, 0.75),
    'F75-100': (0.75, 1.0)
}

print("Loading allocation functions...")
with open(alloc_path, 'rb') as f:
    allocation_funcs = pickle.load(f)

inv_alloc = {}
for cls_name, alloc in allocation_funcs.items():
    xp = alloc['cum_median']
    fp = alloc['thresholds']
    inv_alloc[cls_name] = lambda target, xp=xp, fp=fp: np.interp(target, xp, fp, left=0.0, right=1.0)

print("Loading historical class dictionary...")
with open(class_dict_path, 'rb') as f:
    class_dict = pickle.load(f)

date_to_class = {}
for cls_name, dates in class_dict.items():
    for d in dates:
        if isinstance(d, pd.Timestamp):
            d = d.date()
        date_to_class[d] = cls_name

print("Computing daily Fdry from combined dataset...")
with open(combined_data_path, 'rb') as f:
    df = pickle.load(f)

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

def compute_fdry(row):
    total = row.count()
    if total == 0:
        return np.nan
    dry = (row <= 0).sum()
    return dry / total

fdry_series = df.apply(compute_fdry, axis=1)
fdry_series.name = 'Fdry'
fdry_dict = {date.date(): value for date, value in fdry_series.items()}
print(f"Computed Fdry for {len(fdry_dict)} days.")

years = range(1951, 2022)
months = range(1, 13)
total_processed = 0
skipped_no_class = 0
skipped_no_fdry = 0
out_of_range_count = 0

for year in years:
    for month in months:
        month_str = f"{year:04d}-{month:02d}"
        input_file = prob_dir / f"occurrence_probability_{month_str}.nc"
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping.")
            continue
        print(f"Processing {month_str}...")
        ds = xr.open_dataset(input_file)
        land_mask = ds['land_mask'].values
        if land_mask.dtype == np.float32:
            land_bool = land_mask > 0.5
        else:
            land_bool = land_mask.astype(bool)
        new_binary = np.full_like(ds['prob_map'].values, np.nan, dtype=np.float32)
        for t_idx, time_val in enumerate(ds.time.values):
            date = pd.to_datetime(time_val).date()
            cls_name = date_to_class.get(date)
            if cls_name is None:
                print(f"  Warning: {date} not found in class dictionary. Setting binary to NaN.")
                skipped_no_class += 1
                continue
            fdry = fdry_dict.get(date)
            if fdry is None or np.isnan(fdry):
                print(f"  Warning: {date} has no valid Fdry. Setting binary to NaN.")
                skipped_no_fdry += 1
                continue
            low, high = expected_ranges[cls_name]
            if not (low <= fdry <= high):
                print(f"  NOTE: {date} class {cls_name} has Fdry = {fdry:.3f} outside [{low}, {high}]")
                out_of_range_count += 1
            threshold = inv_alloc[cls_name](fdry)
            prob_slice = ds['prob_map'].isel(time=t_idx).values
            binary_slice = (prob_slice > threshold).astype(np.float32)
            binary_slice[~land_bool] = np.nan
            new_binary[t_idx, :, :] = binary_slice
            total_processed += 1
        ds['binary_map'] = (('time', 'y', 'x'), new_binary)
        ds['binary_map'].attrs.update({
            'long_name': 'Binary rainfall occurrence',
            'description': 'Generated using dynamic thresholds from allocation functions',
            'units': 'binary',
            'valid_range': [0, 1]
        })
        ds.attrs['history'] = ds.attrs.get('history', '') + f'\nBinary maps updated with allocation functions on {pd.Timestamp.now()}'
        ds.attrs['binary_method'] = 'Dynamic cutoff via allocation functions (calibrated 2026-02-15)'
        output_file = output_dir / f"binary_occurrence_{month_str}.nc"
        ds.to_netcdf(output_file)
        ds.close()
        print(f"  Saved {output_file}")

print("\nProcessing completed.")
print(f"Total days processed: {total_processed}")
print(f"Days skipped (no class): {skipped_no_class}")
print(f"Days skipped (no Fdry): {skipped_no_fdry}")
print(f"Days with Fdry outside class range: {out_of_range_count}")

base_dir = Path("outputs/reconstruction_IX")
class_dict_path = base_dir / "class_dict_ALL.pkl"
classic_dir = base_dir / "Phase_I/Binary_Occurrence_Grids_classic from folder III"
new_dir = base_dir / "Phase_I/Binary_Occurrence_Grids"
output_dir = new_dir
output_dir.mkdir(parents=True, exist_ok=True)

with open(class_dict_path, 'rb') as f:
    class_dict = pickle.load(f)
date_to_class = {}
for cls, dates in class_dict.items():
    for d in dates:
        if isinstance(d, pd.Timestamp):
            d = d.date()
        date_to_class[d] = cls

def get_all_dates_from_monthly_files(directory):
    dates = set()
    for file_path in directory.glob("binary_occurrence_*.nc"):
        try:
            with xr.open_dataset(file_path) as ds:
                file_dates = pd.to_datetime(ds.time.values).date
                dates.update(file_dates)
        except Exception as e:
            print(f"Warning: could not read {file_path}: {e}")
    return dates

print("Scanning classic binary maps...")
classic_dates = get_all_dates_from_monthly_files(classic_dir)
print(f"Found {len(classic_dates)} dates in classic dataset.")
print("Scanning new binary maps...")
new_dates = get_all_dates_from_monthly_files(new_dir)
print(f"Found {len(new_dates)} dates in new dataset.")
common_dates = sorted(classic_dates & new_dates)
print(f"Common dates: {len(common_dates)}")
if len(common_dates) < 5:
    raise RuntimeError("Less than 5 common dates available for comparison.")
selected_dates = random.sample(common_dates, 5)
print("Selected dates:", [d.strftime('%Y-%m-%d') for d in selected_dates])

def get_slice_from_monthly(date, directory):
    year_month = date.strftime('%Y-%m')
    file_path = directory / f"binary_occurrence_{year_month}.nc"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")
    with xr.open_dataset(file_path) as ds:
        time_vals = pd.to_datetime(ds.time.values).date
        idx = np.where(time_vals == date)[0]
        if len(idx) == 0:
            raise ValueError(f"Date {date} not found in {file_path}")
        t = idx[0]
        slice_data = ds.binary_map.isel(time=t).values
        land_mask = ds.land_mask.values if 'land_mask' in ds else None
    return slice_data, land_mask

n_dates = len(selected_dates)
fig, axes = plt.subplots(n_dates, 2, figsize=(12, 4 * n_dates))
if n_dates == 1:
    axes = axes.reshape(1, -1)
for i, date in enumerate(selected_dates):
    classic_slice, _ = get_slice_from_monthly(date, classic_dir)
    new_slice, _ = get_slice_from_monthly(date, new_dir)
    cls = date_to_class.get(date, 'Unknown')
    ax = axes[i, 0]
    im0 = ax.imshow(classic_slice, cmap='RdBu', vmin=0, vmax=1, aspect='auto', origin='lower')
    ax.set_title(f"Classic binary map (static threshold)\n{date} (Class: {cls})", fontsize=10)
    ax.axis('off')
    ax = axes[i, 1]
    im1 = ax.imshow(new_slice, cmap='RdBu', vmin=0, vmax=1, aspect='auto', origin='lower')
    ax.set_title(f"New binary map (allocation function)\n{date}", fontsize=10)
    ax.axis('off')
plt.tight_layout()
comp_plot_path = base_dir / "before_after_comparison_historical IV.png"
plt.savefig(comp_plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Comparison plot saved to {comp_plot_path}")
