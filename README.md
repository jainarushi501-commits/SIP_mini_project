# ForenSight — Non-ML Digital Image Forensics

ForenSight is an open-source, deterministic digital image forensics and manipulation detection platform.

It is built strictly using classical image processing, signal processing, spatial statistical analysis, and cryptographic provenance verification.

## 1. Absolute Scientific Integrity Principles
- **Zero Machine Learning / Deep Learning:** No neural networks, pretrained classifiers, or generative AI detectors.
- **No Fabricated Outputs:** All scores, metrics, heatmaps, and bounding boxes stem directly from verified algorithmic execution.
- **Realistic Conclusions:** The system never equates "no anomalies detected" with "genuine image."
- **Inconclusive & Applicability Handling:** Supports `INCONCLUSIVE`, `NOT_APPLICABLE`, and `FAILED` states transparently.

---

## 2. Implemented Forensic Modules & Citations

1. **Metadata & File Characteristic Inspection**
   - Extracts EXIF tags, camera make/model, timestamps, and known photo editing software signatures (Photoshop, GIMP, Canva, Lightroom).
   - *Citation:* ISO 12234-2: TIFF/EP image data format; Kee, Johnson & Farid (2011).

2. **Error Level Analysis (ELA)**
   - Computes per-channel recompression differences at a documented quality ($Q=90$), generates calibrated difference heatmaps, and flags localized statistical outliers ($> \mu + 2.5\sigma$).
   - *Citation:* Krawetz, N. (2007). *A picture's worth...*; Farid, H. (2009). *Image forgery detection*.

3. **2D Fast Fourier Transform (FFT) Frequency Analysis**
   - Applies 2D Hann windowing to suppress boundary leakage, calculates centered log-magnitude spectrum $S(u, v) = \log(1 + |F(u, v)|)$, measures band energy, and clusters conjugate periodic peaks.
   - *Citation:* Popescu, A. C., & Farid, H. (2005). *Exposing digital forgeries by detecting traces of resampling*. IEEE TSP.

4. **Resampling Spectral Correlation**
   - Calculates 2nd-order discrete differences $[-1, 2, -1]$, computes 1D marginal power spectrum, and measures peak-to-average ratio (PAR) for linear interpolation periodicities.
   - *Citation:* Li, Y., et al. (2026). *Image Resampling Detection via Spectral Correlation with False Alarm Control*. PubMed ID: 42574416.

5. **Copy-Move Duplication Analysis**
   - Extracts local SIFT keypoints, conducts in-image Lowe's 2NN ratio test with strict minimum spatial separation ($\ge 30\text{ px}$), clusters matches via RANSAC homography, and localizes duplicated regions.
   - *Citation:* Amerini, I., et al. (2011). *A SIFT-based forensic method for copy-move attack detection*. IEEE TIFS.

6. **Noise Residual & High-Pass Dispersion Analysis**
   - Extracts sensor noise residual via median subtraction, calculates patch-wise Median Absolute Deviation ($\sigma = \text{MAD} / 0.6745$), and maps localized $z$-score deviations.
   - *Citation:* Lyu, S., Pan, X., & Zhang, X. (2014). *Exposing image splicing with inconsistent local noise levels*. IEEE TIFS.

7. **Edge & Boundary Discontinuity Analysis**
   - Measures Sobel gradient magnitude, Canny edge density, and checks for abnormally sharp closed-loop boundary seams.
   - *Citation:* Canny, J. (1986). *A computational approach to edge detection*. IEEE PAMI.

8. **JPEG Quantization & 8x8 DCT Block Analysis**
   - Inspects DQT matrices, blocking artifact index across 8-pixel boundaries, and AC coefficient histogram comb structures for double-compression traces.
   - *Citation:* Lukas, J., & Fridrich, J. (2003); Bianchi, T., & Piva, A. (2012).

9. **C2PA Content Credentials & Cryptographic Provenance**
   - Detects JUMBF boxes and delegates to `c2patool` for cryptographic asset binding verification.
   - *Citation:* C2PA Specification v1.3 (https://spec.c2pa.org/specifications/).

---

## 3. CG-1050 Dataset Integration & Benchmark

The primary benchmark is the **CG-1050** research dataset:
- *Reference:* Maikol Castro, Dora M. Ballesteros, Diego Renza, *"A dataset of 1050-tampered color and grayscale images (CG-1050),"* Data in Brief, DOI: `10.1016/j.dib.2019.104864`.
- *Official Mendeley Download:* https://data.mendeley.com/datasets/dk84bmnyw9/2
- *Kaggle Companion:* https://www.kaggle.com/datasets/saurabhshahane/cg1050

### Dataset Installation
1. Download the CG-1050 dataset archive.
2. Unpack into `data/CG-1050/` or set a custom path in `.env`:
   ```bash
   FORENSIGHT_DATASET_DIR=/path/to/CG-1050
   ```
3. Inspect and validate image pairing:
   ```bash
   source venv/bin/activate
   python scripts/validate_dataset.py
   ```

### Evaluation Protocol
- **Source-Grouped Partitioning:** Derivatives of the same original source photograph are strictly isolated to either the calibration set or the held-out test set (zero cross-partition leakage).
- **Frozen Thresholds:** Thresholds are tuned on calibration data only, and completely frozen before running evaluation on the held-out test set.
- **Target Accuracy:** $\ge 80\%$ image-level accuracy on supported manipulation tasks is a metric to measure and report honestly, not assume.

To execute the benchmark:
```bash
python scripts/run_benchmark.py --dataset_dir data/CG-1050 --test_ratio 0.3 --seed 42
```

---

## 4. Setup & Launch Instructions

### Requirements
- macOS / Linux
- Python 3.11, 3.12, or 3.13

### Step-by-Step Launch
1. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run automated test suite:
   ```bash
   PYTHONPATH=. pytest -v
   ```

3. Launch ForenSight application server:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

---

## 5. REST API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | System health, version, and non-ML architecture confirmation |
| `/api/capabilities` | GET | Forensic module parameters and supported formats |
| `/api/analyze` | POST | Uploads image and executes complete deterministic forensic pipeline |
| `/api/analysis/{id}` | GET | Fetches JSON analysis report for given analysis session |
| `/api/analysis/{id}/pdf` | GET | Downloads publication-quality forensic PDF summary report |
| `/api/dataset/status` | GET | Returns validation status and pair count of CG-1050 dataset |
| `/api/benchmark/run` | POST | Executes source-grouped held-out benchmark and returns metrics |
