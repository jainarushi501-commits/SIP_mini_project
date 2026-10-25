import io
import numpy as np
from PIL import Image
import cv2
from scipy.fftpack import dct
from app.schemas import ModuleResult, ModuleStatus

def analyze_jpeg_dct(img: Image.Image, raw_bytes: bytes, original_format: str) -> ModuleResult:
    """
    Forensic Algorithm H: JPEG Quantization Table & 8x8 DCT Block Analysis.
    Extracts quantization tables, investigates primary/secondary compression histogram modes,
    and analyzes blocking artifact grid alignment.
    """
    citations = [
        "Lukas, J., & Fridrich, J. (2003). Estimation of primary quantization matrix in double compressed JPEG images. Digital Forensics.",
        "Bianchi, T., & Piva, A. (2012). Image forgery localization via block-grained analysis of JPEG artifacts. IEEE TIFS."
    ]
    limitations = [
        "Only applicable to JPEG images where quantization tables and block boundaries are preserved.",
        "Non-standard quantization tables may simply reflect custom camera firmware, not tampering.",
        "Cropping images misaligns the 8x8 DCT grid, which requires grid shift estimation."
    ]
    confounding_factors = [
        "Resaving an image at quality 100 or using custom web optimizers (MozJPEG) alters standard quantization expectations."
    ]

    if original_format.upper() not in ["JPEG", "JPG"]:
        return ModuleResult(
            module_name="JPEG Block & DCT Analysis",
            status=ModuleStatus.NOT_APPLICABLE,
            verdict_contribution="NOT_APPLICABLE",
            measurements={"format": original_format, "reason": "Requires lossy JPEG image with DCT quantization data."},
            thresholds={},
            candidate_regions=[],
            visualizations={},
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )

    try:
        # Extract quantization tables via PIL if present
        quant_tables = getattr(img, "quantization", {})
        has_quant = bool(quant_tables)
        
        # Format tables into readable lists
        clean_q_tables = {}
        if has_quant:
            for k, table in quant_tables.items():
                clean_q_tables[f"table_{k}"] = [int(v) for v in table] if isinstance(table, (list, tuple)) else str(table)

        # Grayscale DCT block analysis (8x8)
        gray = np.array(img.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # Measure 8x8 blocking artifact strength along 8-pixel boundaries
        # Blocking artifact index (BAI): compare difference at boundary (modulo 8 == 0) vs inside blocks
        if h >= 16 and w >= 16:
            diff_h = np.abs(gray[1:, :] - gray[:-1, :])
            diff_w = np.abs(gray[:, 1:] - gray[:, :-1])

            # Horizontal boundary differences
            h_boundary = diff_h[7::8, :]
            h_internal = np.delete(diff_h, slice(7, None, 8), axis=0)

            # Vertical boundary differences
            w_boundary = diff_w[:, 7::8]
            w_internal = np.delete(diff_w, slice(7, None, 8), axis=1)

            boundary_mean = float((np.mean(h_boundary) + np.mean(w_boundary)) / 2.0)
            internal_mean = float((np.mean(h_internal) + np.mean(w_internal)) / 2.0)
            blocking_ratio = float(boundary_mean / (internal_mean + 1e-7))
        else:
            blocking_ratio = 1.0

        # DCT coefficient histogram periodicity check (indicative of double JPEG compression)
        # Partition into 8x8 blocks, compute 2D DCT for each
        blocks_h = h // 8
        blocks_w = w // 8
        ac_coeffs = []

        # Sample up to 1000 blocks to keep analysis real-time
        max_blocks = min(1000, blocks_h * blocks_w)
        step = max(1, (blocks_h * blocks_w) // max_blocks)
        
        block_count = 0
        for i in range(0, blocks_h):
            for j in range(0, blocks_w):
                block_count += 1
                if block_count % step != 0:
                    continue
                block = gray[i*8:(i+1)*8, j*8:(j+1)*8] - 128.0
                # 2D DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                # Grab low-frequency AC coefficient (0, 1) and (1, 0)
                ac_coeffs.append(dct_block[0, 1])
                ac_coeffs.append(dct_block[1, 0])

        ac_coeffs = np.array(ac_coeffs)
        
        # Check histogram of AC coefficients for periodic zero-valleys or double compression combs
        hist, bin_edges = np.histogram(ac_coeffs, bins=60, range=(-30, 30))
        # Measure comb factor (variation between adjacent bins)
        hist_diff = np.abs(hist[1:] - hist[:-1])
        comb_score = float(np.std(hist_diff) / (np.mean(hist_diff) + 1e-5))

        # Decision
        double_compression_suspected = (comb_score > 2.8) and (blocking_ratio > 1.15)
        contribution = "DOUBLE_COMPRESSION_CHARACTERISTICS" if double_compression_suspected else "CONSISTENT_SINGLE_COMPRESSION"

        measurements = {
            "has_quantization_tables": has_quant,
            "quantization_tables_count": len(clean_q_tables),
            "blocking_artifact_ratio": round(blocking_ratio, 3),
            "dct_histogram_comb_score": round(comb_score, 3),
            "double_compression_suspected": bool(double_compression_suspected),
            "quantization_tables": clean_q_tables
        }

        return ModuleResult(
            module_name="JPEG Block & DCT Analysis",
            status=ModuleStatus.COMPLETED,
            verdict_contribution=contribution,
            measurements=measurements,
            thresholds={
                "blocking_artifact_ratio_threshold": 1.15,
                "dct_comb_score_threshold": 2.8
            },
            candidate_regions=[],
            visualizations={},
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
    except Exception as e:
        return ModuleResult(
            module_name="JPEG Block & DCT Analysis",
            status=ModuleStatus.FAILED,
            error_message=str(e),
            limitations=limitations,
            confounding_factors=confounding_factors,
            citations=citations
        )
