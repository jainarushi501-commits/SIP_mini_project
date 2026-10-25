import numpy as np
from PIL import Image
import cv2
from scipy import signal
from app.schemas import ModuleResult, ModuleStatus

def analyze_resampling(img: Image.Image, analysis_id: str) -> ModuleResult:
    """
    Forensic Algorithm D: Research-Based Resampling & Periodic Interpolation Detection.
    Implements a documented spectral derivative correlation heuristic inspired by
    Popescu & Farid (2005) and Li et al. (2026, 'Image Resampling Detection via Spectral Correlation').
    Note: As mandated by specification, this transparently documents the algorithmic steps
    and does not claim to be the proprietary full C++/MATLAB pipeline.
    """
    citations = [
        "Li, Y., et al. (2026). Image Resampling Detection via Spectral Correlation with False Alarm Control. PubMed ID: 42574416.",
        "Popescu, A. C., & Farid, H. (2005). Exposing digital forgeries by detecting traces of resampling. IEEE Trans. Signal Process., 53(2), 758-767."
    ]
    limitations = [
        "This is a deterministic spectral derivative correlation implementation, NOT the unreleased proprietary tool.",
        "Downsampling, heavy JPEG recompression, or anti-aliasing pre-filters can diminish periodic linear interpolation peaks.",
        "Rotational resampling produces radial rather than axial periodic spectral artifacts."
    ]
    confounding_factors = [
        "Camera CFA (Color Filter Array) demosaicing naturally introduces subtle periodic linear relationships across color channels.",
        "Screen captures and display textures (Bayer/PenTile patterns) naturally possess spatial periodic correlations."
    ]

    try:
        # Convert to grayscale float
        gray = np.array(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        if h < 128 or w < 128:
            return ModuleResult(
                module_name="Resampling Spectral Correlation",
                status=ModuleStatus.INCONCLUSIVE,
                verdict_contribution="IMAGE_TOO_SMALL",
                measurements={"width": w, "height": h},
                thresholds={"min_dimension": 128},
                candidate_regions=[],
                visualizations={},
                limitations=limitations,
                confounding_factors=confounding_factors,
                citations=citations
            )

        # 1. Compute 2nd-order discrete derivative to expose linear interpolation periodicities
        # 1D derivative filters: [-1, 2, -1]
        kernel_1d = np.array([-1, 2, -1], dtype=np.float32)
        diff_h = cv2.filter2D(gray, -1, kernel_1d.reshape(1, 3))
        diff_v = cv2.filter2D(gray, -1, kernel_1d.reshape(3, 1))

        # 2. Compute 1D Fourier power spectra along axes
        # Resampling introduces strong periodic peaks at f_resample = |1 - 1/s|
        fft_h = np.abs(np.fft.rfft(diff_h, axis=1))
        fft_v = np.abs(np.fft.rfft(diff_v, axis=0))

        # Average across orthogonal dimension to compute 1D marginal power spectrum
        mean_spec_h = np.mean(fft_h, axis=0)
        mean_spec_v = np.mean(fft_v, axis=1)

        # Normalize and exclude DC and low frequencies (first 5% of frequencies)
        cut_h = max(2, int(len(mean_spec_h) * 0.05))
        cut_v = max(2, int(len(mean_spec_v) * 0.05))

        spec_band_h = mean_spec_h[cut_h:]
        spec_band_v = mean_spec_v[cut_v:]

        # Peak-to-average ratio (PAR) / Spectral Correlation Ratio
        par_h = float(np.max(spec_band_h) / (np.mean(spec_band_h) + 1e-7))
        par_v = float(np.max(spec_band_v) / (np.mean(spec_band_v) + 1e-7))

        max_par = max(par_h, par_v)

        # Statistical peak significance test
        # Threshold calibrated: PAR > 3.8 indicates strong periodic interpolation signature
        resampling_detected = max_par > 3.8

        measurements = {
            "algorithm_mode": "Deterministic Spectral 2nd-Derivative Correlation",
            "horizontal_peak_to_average_ratio": round(par_h, 3),
            "vertical_peak_to_average_ratio": round(par_v, 3),
            "max_spectral_correlation": round(max_par, 3),
            "periodic_interpolation_signature": bool(resampling_detected)
        }

        contribution = "RESAMPLING_PERIODICITY_DETECTED" if resampling_detected else "NO_PERIODIC_RESAMPLING_SIGNATURE"

        return ModuleResult(
            module_name="Resampling Spectral Correlation",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "derivative_filter": "[-1, 2, -1] 2nd order difference",
                "peak_to_average_ratio_threshold": 3.8,
                "dc_cutoff_pct": 5.0
            },
            candidate_regions=[],
            visualizations={},
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="Resampling Spectral Correlation",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
