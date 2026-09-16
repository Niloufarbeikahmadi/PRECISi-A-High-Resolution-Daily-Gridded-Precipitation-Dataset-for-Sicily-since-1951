# PRECISi: A High-Resolution Daily Gridded Precipitation Dataset for Sicily (1951–2022)

This repository contains the scientific code used to generate the PRECISi dataset, a high-resolution daily gridded precipitation dataset for Sicily (1951–2022). The dataset is derived from a doubly conditional geostatistical framework combining rainfall occurrence and rainfall magnitude modelling.

Manuscript: `essd-2026-679`  
Title: *PRECISi: A High-Resolution Daily Gridded Precipitation Dataset for Sicily (1951–2022) Derived from a Doubly Conditional Geostatistical Framework*  
Authors: Niloufar Beikahmadi et al.  
Journal: Earth System Science Data (ESSD)

## Repository structure

```text
src/
  01_occurrence_calibration.py
  02_magnitude_calibration_I2.py
  03_magnitude_calibration_I3.py
  04_occurrence_reconstruction.py
  05_magnitude_reconstruction_I2.py
  06_magnitude_reconstruction_I3.py
  07_conditioning_I2_binarization.py
  08_bias_correction_I2.py
  09_bias_correction_I3.py

docs/
  input_output.md
  workflow.md

data/
  README.md
```
Data and code availability
The source code is provided in this repository.
The input station data, benchmark grids, DEM, and generated NetCDF products are not stored in this repository because of size and licensing constraints. They are archived at:

DOI: [to be added]

Repository: [Zenodo/figshare to be added]

Installation
```text
conda create -n precipi python=3.10
conda activate precipi
pip install -r requirements.txt
```
Usage
Place the input files in data/input/ and benchmark data in data/benchmark/.
Then run the scripts in order from src/.

Example:
```text
python src/01_occurrence_calibration.py
python src/02_magnitude_calibration_I2.py
python src/03_magnitude_calibration_I3.py
python src/04_occurrence_reconstruction.py
python src/05_magnitude_reconstruction_I2.py
python src/06_magnitude_reconstruction_I3.py
python src/07_conditioning_I2_binarization.py
python src/08_bias_correction_I2.py
python src/09_bias_correction_I3.py
```

