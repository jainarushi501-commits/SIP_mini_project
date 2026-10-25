# CG-1050 Research Dataset Setup Guide for ForenSight

## Overview
CG-1050 is a benchmark dataset composed of authentic and manipulated image pairs with ground-truth masks.
- Citation: Maikol Castro, Dora M. Ballesteros, Diego Renza, "A dataset of 1050-tampered color and grayscale images (CG-1050)," Data in Brief, DOI: 10.1016/j.dib.2019.104864.
- Official Mendeley repository: https://data.mendeley.com/datasets/dk84bmnyw9/2
- Kaggle companion: https://www.kaggle.com/datasets/saurabhshahane/cg1050

## Installation Steps
1. Download the CG-1050 archive from the Mendeley or Kaggle repository.
2. Unpack the images into this directory (`data/CG-1050`) or any custom local directory.
3. Configure the directory in your `.env` file:
   ```bash
   FORENSIGHT_DATASET_DIR=/absolute/path/to/CG-1050
   ```
4. Verify pairing and integrity using the validation script:
   ```bash
   source venv/bin/activate
   python scripts/validate_dataset.py
   ```
5. Run the reproducible held-out evaluation benchmark:
   ```bash
   python scripts/run_benchmark.py
   ```
