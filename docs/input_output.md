# Input and output data

## Input data

| Item | File | Format | Notes |
|---|---|---|---|
| Historical daily rainfall | `combined_dataset.pkl` | Pickle DataFrame | Daily rainfall, 1951–2022 |
| Station metadata | `merged_rainfall_metadata.csv` | CSV | ID, x, y, elevation |
| Day classification | `final_group_days_ALL.pkl` | Pickle dict | Maps each day to a rainfall class |
| Variogram parameters | `variogram_parameters_summary.csv` | CSV | Magnitude variogram parameters |
| Occurrence models | `class_info_27.11.2025.pkl` | Pickle | Best occurrence models and cutoffs |
| DEM | `dem.nc` | NetCDF | Elevation and land mask |
| Monthly benchmark | `monthly_rainfall.xlsx` | Excel | Monthly benchmark totals |
| Monthly benchmark maps | `MONTHLY_ALTLAS/resampled/P_mens_YYYY_MM.nc` | NetCDF | Resampled benchmark grids |

## Output data

| Item | File pattern | Format | Notes |
|---|---|---|---|
| Occurrence probability | `occurrence_probability_YYYY-MM.nc` | NetCDF | Probability maps |
| Binary occurrence | `binary_occurrence_YYYY-MM.nc` | NetCDF | 0/1 maps |
| Rainfall magnitude | `rainfall_magnitude_YYYY-MM.nc` | NetCDF | mm/day |
| Kriging variance | `kriging_variance_YYYY-MM.nc` | NetCDF | mm² |
| IM product | `im_product_YYYY-MM.nc` | NetCDF | Probability × magnitude |
| Monthly totals | `monthly_rainfall_YYYY-MM.nc` | NetCDF | Monthly aggregated totals |
| Bias-adjusted maps | `BiasAdjusted Maps/*.nc` | NetCDF | After monthly correction |

## Grid

- CRS: EPSG:32633 (UTM Zone 33N)
- Grid spacing: 2000 m
- X range: ~231,300 m to ~588,500 m
- Y range: ~4,044,000 m to ~4,263,000 m
- Land mask: 1 = land, 0 = sea
