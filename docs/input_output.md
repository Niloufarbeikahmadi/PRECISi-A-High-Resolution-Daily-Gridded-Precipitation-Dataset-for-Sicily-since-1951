# Input and output data

## Input data (overview)

The scripts expect the following inputs to be provided locally by the user:

- A daily rainfall archive of station observations for Sicily (`combined_dataset.pkl`).
- Station metadata with UTM Zone 33N coordinates and elevation (`merged_rainfall_metadata.csv`).
- A day classification dictionary mapping each calendar day to a hydrometeorological regime (`final_group_days_ALL.pkl`).
- A summary of calibrated magnitude variogram parameters per regime (`variogram_parameters_summary.csv`).
- A dictionary of calibrated occurrence models per intermittency class (`class_info.pkl`).
- A digital elevation model on the 2 km target grid (`dem.nc`).
- Monthly benchmark precipitation totals for validation (`monthly_rainfall.xlsx`).
- Resampled benchmark monthly grids (`P_mens_YYYY_MM.nc`).

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
- X range: approximately 231,300 m to 588,500 m
- Y range: approximately 4,044,000 m to 4,263,000 m
- Land mask: 1 = land, 0 = sea
