# PRECISi — Scientific Code

Code accompanying the paper:

Beikahmadi, N. et al. (2026). *PRECISi: A High-Resolution Daily Gridded Precipitation Dataset for Sicily (1951–2022) Derived from a Doubly Conditional Geostatistical Framework.* Earth System Science Data.

This repository contains the scientific scripts used to calibrate, reconstruct, and bias-correct the PRECISi daily gridded precipitation dataset for Sicily. The code is shared for methodological clarity and reproducibility. It is not distributed as a package.

## Code organization

All scripts are in `src/` and are intended to be read in the following order:

- `01_occurrence_calibration.py` — Phase I calibration of the rainfall occurrence model (indicator kriging, class-specific variograms, LOOCV).
- `02_magnitude_calibration_I2.py` — Phase II calibration of the rainfall magnitude model, method I2-IIA (Framework IIA, geostatistical).
- `03_magnitude_calibration_I3.py` — Phase II calibration of the rainfall magnitude model, method I3-IIA (Framework IIA, geostatistical).
- `04_occurrence_reconstruction.py` — Historical occurrence reconstruction over 1951–2022 using the calibrated occurrence models.
- `05_magnitude_reconstruction_I2.py` — Historical magnitude reconstruction over 1951–2022, method I2-IIA.
- `06_magnitude_reconstruction_I3.py` — Historical magnitude reconstruction over 1951–2022, method I3-IIA.
- `07_conditioning_I2_binarization.py` — ECDF-based dynamic binarization used by conditioning method I2, plus its historical application.
- `08_bias_correction_I2.py` — Monthly climatological bias correction, monthly aggregation, and benchmark comparison for method I2-IIA.
- `09_bias_correction_I3.py` — Monthly climatological bias correction, monthly aggregation, and benchmark comparison for method I3-IIA.

## Scientific framework (summary)

The PRECISi dataset is built on a doubly conditional geostatistical framework that separates rainfall occurrence from rainfall magnitude:

- **Phase I — Occurrence.** Each day is assigned to one of three intermittency classes (Widespread, Intermediate, Localized) based on the fraction of dry gauges. Indicator kriging is used to estimate the probability of rainfall within each class. Three conditioning strategies are compared: fixed class-specific threshold (I1), dynamic daily ECDF threshold (I2), and probabilistic hurdle weighting (I3).
- **Phase II — Magnitude.** Wet-day rainfall intensity is modelled within nine hydrometeorological regimes obtained by crossing the three intermittency classes with three magnitude superclasses (Light, Moderate, Heavy). A regime-specific quantile transform is applied to wet-station observations, followed by Ordinary Kriging in Gaussian space (Framework IIA). A Regression-Kriging alternative (Framework IIB) using atmospheric and physiographic covariates is also implemented for comparison.
- **Historical reconstruction.** Calibrated parameters are transferred unchanged to the merged AdB–SIAS gauge archive covering 1951–2022 (Transfer-Informed Modelling).
- **Climatological adjustment.** A grid-cell × calendar-month ratio-of-means corrector against an independent monthly benchmark removes the systematic smooth-support bias while leaving interannual variability free.

See `docs/input_output.md` for the input requirements and the structure of the produced NetCDF products.

## Data

The input station records, benchmark grids, DEM, and the generated NetCDF products are **not** distributed with this repository. Only the scientific code is shared. Inputs must be provided locally by the user and outputs are written under `outputs/`; see `docs/input_output.md` for details.

## Dependencies

Python 3.10 or newer, with the packages listed in `requirements.txt` (numpy, pandas, xarray, scipy, gstools, pykrige, scikit-learn, geopandas, shapely, pyproj, hydroeval, matplotlib, plotly, tqdm, netCDF4, openpyxl).

## Contact

Niloufar Beikahmadi — Niloufar.Beikahmadi@gmail.com  

## License

license MIT
