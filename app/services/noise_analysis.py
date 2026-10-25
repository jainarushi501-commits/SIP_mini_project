import numpy as np
from PIL import Image
import cv2
from scipy import signal
from app.config import settings
from app.schemas import ModuleResult, ModuleStatus

def analyze_noise(img: Image.Image, analysis_id: str, patch_size: int = 64) -> ModuleResult:
    """
    Forensic Algorithm F: Noise and High-Pass Residual Inconsistency Analysis.
    Extracts sensor noise residual using a median/wavelet high-pass filter,
    calculates robust regional dispersion (MAD - Median Absolute Deviation),
    and identifies candidate spliced or spliced-in patches with anomalous noise variances.
    """
    citations = [
        "Lyu, S., Pan, X., & Zhang, X. (2014). Exposing image splicing with inconsistent local noise levels. IEEE TIFS.",
        "Fridrich, J., & Kodovsky, J. (2012). Rich models for steganalysis of digital images. IEEE TIFS, 7(3), 868-882."
    ]
    limitations = [
        "Regional noise variations can arise naturally from camera depth of field (bokeh), motion blur, and ISO grain in dark areas.",
        "Heavy JPEG compression attenuates high-frequency noise and introduces blocking artifacts that bias residual estimators.",
        "Does not identify specific sensor fingerprints without camera calibration data."
    ]
    confounding_factors = [
        "Uneven scene illumination (e.g., shadows vs direct sunlight) creates varying Poisson photon shot noise levels.",
        "Smooth skies vs dense foliage have radically different texture masking effects."
    ]

    try:
        # Convert to float grayscale
        gray = np.array(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        if h < patch_size * 2 or w < patch_size * 2:
            return ModuleResult(
                module_name="Noise Residual Analysis",
                status=ModuleStatus.INCONCLUSIVE,
                verdict_contribution="IMAGE_TOO_SMALL_FOR_PATCHING",
                measurements={"width": w, "height": h, "patch_size": patch_size},
                thresholds={"min_dimension": patch_size * 2},
                candidate_regions=[],
                visualizations={},
                limitations=limitations,
                confounding_factors=confounding_factors,
                citations=citations
            )

        # 1. High-pass residual extraction via Laplacian / Median subtraction
        # Denoise using 3x3 median filter to estimate the noiseless underlying scene
        denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        residual = gray - denoised

        # 2. Patch-wise robust noise estimation using Median Absolute Deviation (MAD)
        # MAD is resilient to sharp edges within patches: sigma = MAD / 0.6745
        patch_vars = []
        grid_h = h // patch_size
        grid_w = w // patch_size
        var_map = np.zeros((grid_h, grid_w), dtype=np.float32)

        for i in range(grid_h):
            for j in range(grid_w):
                py = i * patch_size
                px = j * patch_size
                patch = residual[py:py+patch_size, px:px+patch_size]
                
                # Robust sigma calculation
                med = np.median(patch)
                mad = np.median(np.abs(patch - med))
                sigma = mad / 0.6745
                var_map[i, j] = sigma
                patch_vars.append(sigma)

        patch_vars = np.array(patch_vars, dtype=np.float32)
        global_median_sigma = float(np.median(patch_vars))
        global_mad_sigma = float(np.median(np.abs(patch_vars - global_median_sigma))) + 1e-6

        # Standardized z-score map of patch noise
        z_scores = np.abs(var_map - global_median_sigma) / global_mad_sigma

        # Identify candidate anomalous patches (e.g. z-score > 3.0)
        anomaly_mask = z_scores > 3.0
        candidate_regions = []
        anom_y, anom_x = np.where(anomaly_mask)
        
        for y_idx, x_idx in zip(anom_y, anom_x):
            bx = int(x_idx * patch_size)
            by = int(y_idx * patch_size)
            candidate_regions.append({
                "bounding_box": [bx, by, patch_size, patch_size],
                "local_noise_sigma": round(float(var_map[y_idx, x_idx]), 3),
                "z_score_deviation": round(float(z_scores[y_idx, x_idx]), 2)
            })

        # Save diagnostic visualization: Normalized residual & Heatmap
        # Scale residual to visible range
        res_vis = np.clip((residual + 15) * (255.0 / 30.0), 0, 255).astype(np.uint8)
        
        # Upsample var_map to original image dimensions for smooth heatmap
        var_map_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_upscaled = cv2.resize(var_map_norm, (w, h), interpolation=cv2.INTER_NEAREST)
        color_heatmap = cv2.applyColorMap(heatmap_upscaled, cv2.COLORMAP_JET)

        # Highlight anomalous patches with red boundary
        for reg in candidate_regions:
            rx, ry, rw, rh = reg["bounding_box"]
            cv2.rectangle(color_heatmap, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

        viz_filename = f"noise_{analysis_id}.jpg"
        viz_path = settings.GENERATED_DIR / viz_filename
        cv2.imwrite(str(viz_path), color_heatmap)

        res_filename = f"residual_{analysis_id}.jpg"
        res_path = settings.GENERATED_DIR / res_filename
        cv2.imwrite(str(res_path), res_vis)

        # Contribution
        anom_ratio = len(candidate_regions) / max(1, len(patch_vars))
        if 0 < anom_ratio < 0.25 and len(candidate_regions) >= 2:
            contribution = "LOCALIZED_NOISE_INCONSISTENCY"
        else:
            contribution = "HOMOGENEOUS_NOISE_DISTRIBUTION"

        measurements = {
            "patch_size": patch_size,
            "total_patches_evaluated": len(patch_vars),
            "global_median_noise_sigma": round(global_median_sigma, 3),
            "global_noise_dispersion_mad": round(global_mad_sigma, 3),
            "anomalous_patches_count": len(candidate_regions),
            "anomalous_patch_percentage": round(anom_ratio * 100, 2)
        }

        return ModuleResult(
            module_name="Noise Residual & Inconsistency Analysis",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "estimator": "Median Absolute Deviation (MAD / 0.6745)",
                "z_score_anomaly_threshold": 3.0,
                "patch_dimension": f"{patch_size}x{patch_size} px"
            },
            candidate_regions=candidate_regions[:20],  # Limit to top 20 for response payload
            visualizations={
                "noise_variance_heatmap": f"/static/generated/{viz_filename}",
                "highpass_residual": f"/static/generated/{res_filename}"
            },
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="Noise Residual & Inconsistency Analysis",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
