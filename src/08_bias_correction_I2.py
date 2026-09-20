### bias correction + aggregation + benchmark comparison for magnitude (Sicily_Rainfall_Reconstruction_I2) ###

import os, numpy as np, xarray as xr, pandas as pd, matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from tqdm import tqdm
import warnings, logging, time, gc, traceback
from datetime import datetime

# ---------- configuration ----------
BENCHMARK_ASC_FOLDER   = r'data/benchmark/MONTHLY_ALTLAS'
RESAMPLED_FOLDER       = os.path.join(BENCHMARK_ASC_FOLDER, 'resampled')
MODEL_SAMPLE_FILE      = r'outputs/reconstruction_I2/aggregated_monthly_totals/monthly_rainfall_1951-01.nc'
MODEL_AGG_FOLDER       = r'outputs/reconstruction_I2/aggregated_monthly_totals'
MAGNITUDE_FOLDER       = r'outputs/reconstruction_I2/Phase_II/Magnitude_Grids'
OUTPUT_CORRECTED_FOLDER= r'outputs/reconstruction_I2/Phase_II/BiasAdjusted Maps'
CORRECTOR_FILE         = r'outputs/reconstruction_I2/monthly_corrector_maps_1951-2022.nc'
BENCHMARK_EXCEL        = r'data/input/monthly_rainfall.xlsx'
BASE_PATH              = Path(r'outputs/reconstruction_I2')
MAGNITUDE_VAR          = 'rainfall_magnitude'
MAG_FILE_PATTERN       = 'rainfall_magnitude_*.nc'
CORRECTED_FILE_PATTERN = 'rainfall_magnitude_*.nc'

os.makedirs(RESAMPLED_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_CORRECTED_FOLDER, exist_ok=True)

# ---------- 1. resample benchmark ASCII to model grid ----------
def read_ascii_to_dataarray(filepath):
    with open(filepath,'r') as f:
        header = {}
        for _ in range(6):
            line = f.readline().strip()
            if not line: continue
            key, val = line.split(None,1)
            header[key.lower()] = float(val) if '.' in val else int(val)
    ncols = int(header['ncols']); nrows = int(header['nrows'])
    cellsize = header['cellsize']; nodata = header.get('nodata_value',-9999)
    data = pd.read_csv(filepath,skiprows=6,sep='\s+',header=None,dtype=np.float32).values
    data = np.where(data==nodata, np.nan, data)
    xll = header['xllcorner']; yll = header['yllcorner']
    x = xll + cellsize/2 + np.arange(ncols)*cellsize
    y = yll + cellsize/2 + (nrows-1-np.arange(nrows))*cellsize
    return xr.DataArray(data, dims=('y','x'), coords={'y':y,'x':x},
                        name='monthly_rainfall_benchmark',
                        attrs={'units':'mm','crs':'EPSG:32633'})

with xr.open_dataset(MODEL_SAMPLE_FILE) as ds:
    x_target = ds.x.values; y_target = ds.y.values

for asc in sorted(glob(os.path.join(BENCHMARK_ASC_FOLDER,'P_mens_*.asc'))):
    da = read_ascii_to_dataarray(asc)
    da_resampled = da.interp(x=x_target, y=y_target, method='linear', kwargs={'fill_value':np.nan})
    out = os.path.join(RESAMPLED_FOLDER, os.path.splitext(os.path.basename(asc))[0]+'.nc')
    da_resampled.to_netcdf(out, encoding={'monthly_rainfall_benchmark':{'zlib':True}})

# ---------- 2. compute monthly corrector maps ----------
all_months = pd.date_range('1951-01','2021-12',freq='MS')
corrector_list, valid_months = [], []

for date in tqdm(all_months, desc='Corrector maps'):
    y, m = date.year, date.month
    mf = os.path.join(MODEL_AGG_FOLDER, f'monthly_rainfall_{y:04d}-{m:02d}.nc')
    bf = os.path.join(RESAMPLED_FOLDER, f'P_mens_{y:04d}_{m:02d}.nc')
    if not os.path.exists(mf) or not os.path.exists(bf): continue
    with xr.open_dataset(mf) as dm, xr.open_dataset(bf) as db:
        corr = xr.where(dm.monthly_rainfall>0,
                        db.monthly_rainfall_benchmark/dm.monthly_rainfall, np.nan)
        corr = corr.expand_dims(time=[date])
    corrector_list.append(corr); valid_months.append(date)

corrector_da = xr.concat(corrector_list, dim='time')
corrector_da.name = 'monthly_corrector'
corrector_da.to_netcdf(CORRECTOR_FILE, encoding={'monthly_corrector':{'zlib':True,'complevel':4}})

# ---------- 3. apply corrector to daily magnitude maps ----------
corrector_ds = xr.open_dataset(CORRECTOR_FILE)
for date in tqdm(valid_months, desc='Correcting daily maps'):
    y,m = date.year, date.month
    mag_file = os.path.join(MAGNITUDE_FOLDER, f'rainfall_magnitude_{y:04d}-{m:02d}.nc')
    if not os.path.exists(mag_file): continue
    with xr.open_dataset(mag_file) as ds_mag:
        mag_var = ds_mag[MAGNITUDE_VAR]
        corr_this = corrector_ds.monthly_corrector.sel(time=date, drop=True)
        mag_corr = mag_var * corr_this
        ds_corr = ds_mag.copy(deep=False)
        ds_corr[MAGNITUDE_VAR] = mag_corr
        ds_corr.attrs['history'] = ds_mag.attrs.get('history','') + f'\nBias-adjusted {pd.Timestamp.now()}'
        out = os.path.join(OUTPUT_CORRECTED_FOLDER, f'rainfall_magnitude_{y:04d}-{m:02d}.nc')
        ds_corr.to_netcdf(out, encoding={MAGNITUDE_VAR:{'zlib':True,'complevel':4}})
corrector_ds.close()

# ---------- 4. aggregate corrected daily maps to monthly totals ----------
class MonthlyAggregator:
    def __init__(self, base_path, magnitude_path, aggregated_dir, var_name):
        self.magnitude_path = Path(magnitude_path)
        self.aggregated_dir = Path(aggregated_dir)
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)
        self.var_name = var_name
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('Aggregator')
    def get_available_months(self):
        files = list(self.magnitude_path.glob('rainfall_magnitude_*.nc'))
        return sorted([f.stem.split('_')[-1] for f in files])
    def aggregate_month(self, month_str):
        out_file = self.aggregated_dir / f'monthly_rainfall_{month_str}.nc'
        if out_file.exists(): return True
        ds = xr.open_dataset(self.magnitude_path / f'rainfall_magnitude_{month_str}.nc')
        da = ds[self.var_name].sum(dim='time', skipna=True)
        if 'land_mask' in ds: da = da.where(ds.land_mask.astype(bool), np.nan)
        da.attrs = {'units':'mm','month':month_str,'description':'Bias-corrected monthly total'}
        ds_out = xr.Dataset({'monthly_rainfall':da})
        ds_out.to_netcdf(out_file, encoding={'monthly_rainfall':{'zlib':True,'complevel':4,'dtype':'float32','_FillValue':-9999.0}})
        ds.close(); del ds, da; gc.collect()
        return True
    def process_all(self, start=1951, end=2021):
        months = [m for m in self.get_available_months() if start<=int(m[:4])<=end]
        for m in tqdm(months, desc='Aggregating'): self.aggregate_month(m)

agg = MonthlyAggregator(BASE_PATH, OUTPUT_CORRECTED_FOLDER,
                        BASE_PATH / 'aggregated_monthly_totals_corrected', MAGNITUDE_VAR)
agg.process_all()

# ---------- 5. extract spatial averages & merge with benchmark ----------
nc_dir = BASE_PATH / 'aggregated_monthly_totals_corrected'
bench_df = pd.read_excel(BENCHMARK_EXCEL)
preds = []
for f in nc_dir.glob('monthly_rainfall_*.nc'):
    ds = xr.open_dataset(f)
    avg = np.nanmean(ds['monthly_rainfall'].values)
    ds.close()
    parts = f.stem.split('_')[-1].split('-')
    preds.append({'time':f"{int(parts[0]):04d}-{int(parts[1]):02d}", 'pred_rain':round(avg,2)})
pred_df = pd.DataFrame(preds).sort_values('time')
merged = bench_df.merge(pred_df[['time','pred_rain']], on='time', how='left')
merged.to_excel(BASE_PATH / 'monthly_rainfall_with_predictions_corrected.xlsx', index=False)

# ---------- 6. scatter plot (monthly & yearly) ----------
df = pd.read_excel(BASE_PATH / 'monthly_rainfall_with_predictions_corrected.xlsx')
df['date'] = pd.to_datetime(df['time']); df['year'] = df['date'].dt.year
df_v = df.dropna(subset=['rain','pred_rain']).copy()
yearly = df_v.groupby('year')[['rain','pred_rain']].sum().reset_index()
mc = df_v[['rain','pred_rain']].corr().iloc[0,1]
yc = yearly[['rain','pred_rain']].corr().iloc[0,1]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,9))
ax1.scatter(df_v['rain'], df_v['pred_rain'], alpha=0.5, s=10, color='blue')
ax1.plot([0,ax1.get_xlim()[1]],[0,ax1.get_ylim()[1]],'r--'); ax1.set_title(f'Monthly (N={len(df_v)}), R²={mc**2:.3f}')
ax1.grid(True,alpha=0.3)
ax2.scatter(yearly['rain'], yearly['pred_rain'], alpha=0.7, s=40, color='green')
ax2.plot([0,ax2.get_xlim()[1]],[0,ax2.get_ylim()[1]],'r--'); ax2.set_title(f'Yearly (N={len(yearly)}), R²={yc**2:.3f}')
ax2.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig(BASE_PATH / 'scatter_comparison_biascorrected.png', dpi=300); plt.show()
