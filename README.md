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
