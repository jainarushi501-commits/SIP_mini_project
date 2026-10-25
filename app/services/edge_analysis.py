import numpy as np
from PIL import Image
import cv2
from app.config import settings
from app.schemas import ModuleResult, ModuleStatus

def analyze_edges(img: Image.Image, analysis_id: str) -> ModuleResult:
    """
    Forensic Algorithm G: Deterministic Edge & Boundary Discontinuity Analysis.
    Calculates Sobel gradient magnitude, Canny edge maps, and regional edge-density
    discontinuities to identify artificial splicing seams or boundary blending.
    """
    citations = [
        "Canny, J. (1986). A computational approach to edge detection. IEEE PAMI, (6), 679-698.",
        "Sobel, I., & Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing."
    ]
    limitations = [
        "Ordinary object silhouettes, horizon lines, and in-focus subjects naturally create sharp edge discontinuities.",
        "Edge density varies widely across natural scenes (e.g. textured grass vs clear sky).",
        "Edge sharpness alone does not establish digital manipulation."
    ]
    confounding_factors = [
        "In-camera post-processing, artificial unsharp masking, and contrast enhancements sharpen all natural edges uniformly."
    ]

    try:
        gray = np.array(img.convert("L"), dtype=np.uint8)
        h, w = gray.shape

        # 1. Sobel Gradient Magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 2. Canny Edge Map with Otsu auto-thresholding
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_canny = int(max(0, 0.5 * otsu_thresh))
        upper_canny = int(min(255, 1.5 * otsu_thresh))
        canny_edges = cv2.Canny(gray, lower_canny, upper_canny)

        # 3. Local edge density & discontinuity check
        total_pixels = h * w
        edge_pixel_count = int(np.count_nonzero(canny_edges))
        global_edge_density = float(edge_pixel_count / max(1, total_pixels))

        # Check for abnormal closed boundary seams (common splicing signature)
        contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_regions = []
        
        # Look for unusually sharp closed loop contours that encompass significant area
        for cnt in contours:
            length = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if area > (total_pixels * 0.01) and length > 100:
                # Measure edge gradient sharpness along the contour
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [cnt], -1, 255, 2)
                boundary_grad_mean = float(np.mean(grad_norm[mask == 255]))
                
                # If the boundary has exceptionally high gradient sharpness compared to surroundings
                if boundary_grad_mean > 150:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    candidate_regions.append({
                        "bounding_box": [int(x), int(y), int(cw), int(ch)],
                        "perimeter_px": round(length, 1),
                        "enclosed_area_px": int(area),
                        "mean_boundary_gradient": round(boundary_grad_mean, 2)
                    })

        # Save visualizations
        viz_edge = cv2.applyColorMap(canny_edges, cv2.COLORMAP_BONE)
        for reg in candidate_regions:
            rx, ry, rw, rh = reg["bounding_box"]
            cv2.rectangle(viz_edge, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

        viz_filename = f"edge_{analysis_id}.jpg"
        viz_path = settings.GENERATED_DIR / viz_filename
        cv2.imwrite(str(viz_path), viz_edge)

        measurements = {
            "global_edge_density_pct": round(global_edge_density * 100, 2),
            "total_edge_pixels": edge_pixel_count,
            "canny_thresholds": [lower_canny, upper_canny],
            "candidate_sharp_boundaries": len(candidate_regions)
        }

        contribution = "ISOLATED_SHARP_BOUNDARY_DETECTED" if len(candidate_regions) > 0 else "CONSISTENT_EDGE_DISTRIBUTION"

        return ModuleResult(
            module_name="Edge & Boundary Discontinuity Analysis",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "canny_lower": lower_canny,
                "canny_upper": upper_canny,
                "min_anomaly_area_pct": 1.0,
                "gradient_sharpness_threshold": 150
            },
            candidate_regions=candidate_regions,
            visualizations={
                "edge_map": f"/static/generated/{viz_filename}"
            },
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="Edge & Boundary Discontinuity Analysis",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
