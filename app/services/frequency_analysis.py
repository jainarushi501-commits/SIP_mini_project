import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from app.config import settings
from app.schemas import ModuleResult, ModuleStatus

def analyze_fft(img: Image.Image, analysis_id: str) -> ModuleResult:
    """
    Forensic Algorithm C: 2D Fast Fourier Transform (FFT) Frequency Analysis.
    Calculates S(u, v) = log(1 + |F(u, v)|), applies Hann windowing,
    detects candidate periodic peaks, and calculates spectral energy distributions.
    """
    citations = [
        "Gallagher, A. C. (2005). Detection of linear and cubic interpolation in JPEG compressed images. Computer Vision and Pattern Recognition.",
        "Popescu, A. C., & Farid, H. (2005). Exposing digital forgeries by detecting traces of resampling. IEEE Transactions on Signal Processing, 53(2), 758-767."
    ]
    limitations = [
        "A frequency spectrum visualization is NOT by itself proof of manipulation.",
        "Natural periodic patterns (brick walls, textiles, fences, window blinds) create genuine prominent spectral peaks.",
        "Windowing reduces boundary discontinuities but attenuates peripheral high frequencies."
    ]
    confounding_factors = [
        "Sharp image edges and rectangular boundaries cause cross-shaped spectral leakage artifacts.",
        "Repetitive architectural elements, fabrics, halftoning, and screen captures introduce strong grid frequencies."
    ]

    try:
        # 1. Convert to grayscale float array
        gray = np.array(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # 2. Documented 2D Hann windowing to suppress boundary leakage
        hann_y = np.hanning(h)
        hann_x = np.hanning(w)
        window = np.outer(hann_y, hann_x)
        windowed_gray = gray * window

        # 3. 2D Discrete Fourier Transform & Centering
        f_transform = np.fft.fft2(windowed_gray)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)

        # Formula: S(u, v) = log(1 + |F(u, v)|)
        log_spectrum = np.log1p(magnitude)

        # Normalize log spectrum to [0, 255] for visualization
        spec_min, spec_max = log_spectrum.min(), log_spectrum.max()
        if spec_max > spec_min:
            norm_spectrum = ((log_spectrum - spec_min) / (spec_max - spec_min) * 255.0).astype(np.uint8)
        else:
            norm_spectrum = np.zeros((h, w), dtype=np.uint8)

        # 4. Energy distribution metrics across frequency bands (Low, Mid, High)
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_radius = np.sqrt(center_y ** 2 + center_x ** 2)
        norm_radius = dist_from_center / (max_radius + 1e-7)

        total_energy = np.sum(magnitude) + 1e-9
        low_freq_mask = norm_radius <= 0.15
        mid_freq_mask = (norm_radius > 0.15) & (norm_radius <= 0.50)
        high_freq_mask = norm_radius > 0.50

        low_energy_pct = float(np.sum(magnitude[low_freq_mask]) / total_energy * 100)
        mid_energy_pct = float(np.sum(magnitude[mid_freq_mask]) / total_energy * 100)
        high_energy_pct = float(np.sum(magnitude[high_freq_mask]) / total_energy * 100)

        # 5. Candidate periodic peak detection (excluding DC center peak)
        # Apply local maximum filter to log spectrum
        local_max = ndimage.maximum_filter(log_spectrum, size=15)
        is_peak = (log_spectrum == local_max) & (norm_radius > 0.05)  # Exclude DC center component
        
        # A defensible peak must exceed regional mean by at least 3.0 standard deviations
        mean_spec = np.mean(log_spectrum[norm_radius > 0.05])
        std_spec = np.std(log_spectrum[norm_radius > 0.05])
        peak_threshold = mean_spec + 3.2 * std_spec

        peak_y, peak_x = np.where(is_peak & (log_spectrum > peak_threshold))
        
        detected_peaks = []
        for py, px in zip(peak_y[:settings.FFT_MAX_PEAKS], peak_x[:settings.FFT_MAX_PEAKS]):
            peak_val = float(log_spectrum[py, px])
            rel_radius = float(norm_radius[py, px])
            # Check for symmetric conjugate peak (u, v) <-> (-u, -v)
            sym_y, sym_x = 2 * center_y - py, 2 * center_x - px
            has_symmetric = (0 <= sym_y < h and 0 <= sym_x < w and is_peak[sym_y, sym_x])
            detected_peaks.append({
                "coord": [int(px), int(py)],
                "frequency_radius": round(rel_radius, 4),
                "magnitude": round(peak_val, 2),
                "has_conjugate_pair": bool(has_symmetric)
            })

        # Save visualization: Spectrum with colormap & marked prominent peaks
        viz_color = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_VIRIDIS)
        for p in detected_peaks:
            cx, cy = p["coord"]
            cv2.circle(viz_color, (cx, cy), 5, (0, 0, 255), 1)

        viz_filename = f"fft_{analysis_id}.jpg"
        viz_path = settings.GENERATED_DIR / viz_filename
        cv2.imwrite(str(viz_path), viz_color)

        # Contribution
        # Periodic grid peaks may suggest resampling or periodic interpolation
        periodic_pairs_count = sum(1 for p in detected_peaks if p["has_conjugate_pair"])
        if periodic_pairs_count >= 4:
            contribution = "PERIODIC_FREQUENCY_ANOMALIES_DETECTED"
        else:
            contribution = "TYPICAL_CONTINUOUS_SPECTRUM"

        measurements = {
            "spectrum_dimensions": [w, h],
            "low_frequency_energy_pct": round(low_energy_pct, 2),
            "mid_frequency_energy_pct": round(mid_energy_pct, 2),
            "high_frequency_energy_pct": round(high_energy_pct, 2),
            "detected_peaks_count": len(detected_peaks),
            "conjugate_peak_pairs_count": periodic_pairs_count,
            "spectral_entropy": round(float(-np.sum((magnitude / total_energy) * np.log(magnitude / total_energy + 1e-12))), 4),
            "peaks": detected_peaks
        }

        return ModuleResult(
            module_name="2D Fourier Frequency Analysis (FFT)",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "window": "Hann (2D outer product)",
                "peak_neighborhood_size": 15,
                "peak_threshold": "mean + 3.2 * std",
                "dc_exclusion_radius": "5% of nyquist"
            },
            candidate_regions=[],
            visualizations={
                "fft_spectrum": f"/static/generated/{viz_filename}"
            },
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="2D Fourier Frequency Analysis (FFT)",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
