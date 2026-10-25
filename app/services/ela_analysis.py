import io
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from app.config import settings
from app.schemas import ModuleResult, ModuleStatus

def analyze_ela(img: Image.Image, original_format: str, analysis_id: str, quality: int = 90, scale: int = 15) -> ModuleResult:
    """
    Forensic Algorithm B: Error Level Analysis (ELA) for JPEG images.
    Resaves image at a documented quality, computes per-channel absolute differences,
    extracts regional statistics, and saves real diagnostic heatmaps.
    """
    citations = [
        "Krawetz, N. (2007). A picture's worth... Hacker Factor Solutions.",
        "Farid, H. (2009). Image forgery detection. IEEE Signal Processing Magazine, 26(2), 16-25."
    ]
    limitations = [
        "ELA is NOT proof of forgery. Highly textured regions, sharp edges, and high-frequency content naturally produce higher error levels.",
        "Previous multi-generation re-saving and uniform compression artifacts alter ELA patterns.",
        "ELA is strictly applicable to lossy JPEG compression and cannot evaluate uncompressed PNGs natively."
    ]
    confounding_factors = [
        "Text, geometric patterns, contrasty borders, and noisy skies naturally exhibit elevated recompression error.",
        "Multiple previous JPEG saves at differing qualities create complex composite error curves."
    ]

    # Format check: ELA is fundamentally designed for JPEG recompression analysis
    if original_format.upper() not in ["JPEG", "JPG"]:
        return ModuleResult(
            module_name="Error Level Analysis (ELA)",
            status=ModuleStatus.NOT_APPLICABLE,
            verdict_contribution="NOT_APPLICABLE",
            measurements={"format": original_format, "reason": "ELA is designed specifically for JPEG compression analysis."},
            thresholds={"quality": quality, "scale": scale},
            candidate_regions=[],
            visualizations={},
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )

    try:
        # Convert image to RGB for standardized comparison
        rgb_img = img.convert("RGB")
        orig_arr = np.array(rgb_img, dtype=np.float32)

        # 1. Recompress in memory at specified JPEG quality
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed_img = Image.open(buf)
        recomp_arr = np.array(recompressed_img, dtype=np.float32)

        # 2. Calculate per-channel absolute difference
        diff = np.abs(orig_arr - recomp_arr)
        
        # 3. Compute real mathematical statistics
        mean_error = float(np.mean(diff))
        median_error = float(np.median(diff))
        std_error = float(np.std(diff))
        p95_error = float(np.percentile(diff, 95))
        max_error = float(np.max(diff))

        # 4. Generate visual difference map with explicit scale factor
        scaled_diff = np.clip(diff * scale, 0, 255).astype(np.uint8)
        
        # Also compute a grayscale intensity map for regional peak localization
        gray_diff = cv2.cvtColor(scaled_diff, cv2.COLOR_RGB2GRAY)
        
        # Find candidate high-discrepancy regions using actual statistical threshold (mean + 3*std)
        anomaly_threshold = mean_error * scale + 2.5 * (std_error * scale)
        _, thresh_mask = cv2.threshold(gray_diff, int(min(250, anomaly_threshold)), 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_regions = []
        h, w = gray_diff.shape
        min_region_area = (h * w) * 0.005  # At least 0.5% of image area to avoid single noisy pixels
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_region_area:
                x, y, rw, rh = cv2.boundingRect(cnt)
                reg_mask = thresh_mask[y:y+rh, x:x+rw]
                reg_mean = float(np.mean(gray_diff[y:y+rh, x:x+rw]))
                candidate_regions.append({
                    "bounding_box": [int(x), int(y), int(rw), int(rh)],
                    "area_pixels": int(area),
                    "mean_error_intensity": round(reg_mean, 2)
                })

        # Save diagnostic visualizations to static generated folder
        viz_filename = f"ela_{analysis_id}.jpg"
        viz_path = settings.GENERATED_DIR / viz_filename
        cv2.imwrite(str(viz_path), cv2.cvtColor(scaled_diff, cv2.COLOR_RGB2BGR))

        # Save colormap heatmap
        heatmap_color = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
        heatmap_filename = f"ela_heat_{analysis_id}.jpg"
        heatmap_path = settings.GENERATED_DIR / heatmap_filename
        cv2.imwrite(str(heatmap_path), heatmap_color)

        measurements = {
            "mean_error": round(mean_error, 4),
            "median_error": round(median_error, 4),
            "std_error": round(std_error, 4),
            "p95_error": round(p95_error, 4),
            "max_error": round(max_error, 4),
            "recompression_quality": quality,
            "display_scale_factor": scale,
            "anomaly_threshold_used": round(anomaly_threshold, 2),
            "candidate_regions_count": len(candidate_regions)
        }

        # Contribution verdict based on statistically distinct localized error discrepancy
        # If max_error > 25 and candidate regions exist with elevated error relative to median
        if len(candidate_regions) > 0 and (p95_error > median_error * 3.5) and median_error > 0.5:
            contribution = "ELEVATED_LOCAL_DISCREPANCY"
        else:
            contribution = "UNIFORM_COMPRESSION_ERROR"

        return ModuleResult(
            module_name="Error Level Analysis (ELA)",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "quality": quality,
                "scale": scale,
                "outlier_factor": "mean + 2.5 * std",
                "min_region_area_pct": 0.5
            },
            candidate_regions=candidate_regions,
            visualizations={
                "ela_diff": f"/static/generated/{viz_filename}",
                "ela_heatmap": f"/static/generated/{heatmap_filename}"
            },
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="Error Level Analysis (ELA)",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
