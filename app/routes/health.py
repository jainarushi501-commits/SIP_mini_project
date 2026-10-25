from fastapi import APIRouter
from app.config import settings

router = APIRouter()

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mode": "deterministic-forensics-only",
        "ml_models_loaded": 0,
        "external_ai_apis": 0
    }

@router.get("/capabilities")
def get_capabilities():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "supported_formats": list(settings.ALLOWED_EXTENSIONS),
        "max_upload_size_mb": settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024),
        "max_pixels": settings.MAX_IMAGE_PIXELS,
        "forensic_modules": [
            {
                "id": "metadata",
                "name": "Metadata & File Characteristic Inspection",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "EXIF parsing, editor signatures, timestamp contradictions, GPS tags."
            },
            {
                "id": "ela",
                "name": "Error Level Analysis (ELA)",
                "supports": ["JPEG"],
                "description": "Recompression error differential map and regional statistical analysis."
            },
            {
                "id": "fft",
                "name": "2D Fast Fourier Transform (FFT) Frequency Analysis",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "Log magnitude spectrum, Hann windowing, conjugate periodic peak detection."
            },
            {
                "id": "resampling",
                "name": "Resampling Spectral Correlation",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "2nd-derivative 1D marginal power spectrum peak-to-average ratio."
            },
            {
                "id": "copy_move",
                "name": "Copy-Move Duplication Analysis",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "SIFT/ORB keypoints with Lowe's 2NN ratio test, spatial distance filtering, RANSAC homography."
            },
            {
                "id": "noise",
                "name": "Noise & High-Pass Residual Analysis",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "Laplacian/median residual extraction and patch-wise MAD dispersion z-score map."
            },
            {
                "id": "edge",
                "name": "Edge & Boundary Discontinuity Analysis",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "Sobel gradients, Canny edge density, sharp boundary seam localization."
            },
            {
                "id": "jpeg_dct",
                "name": "JPEG Quantization & 8x8 DCT Analysis",
                "supports": ["JPEG"],
                "description": "Quantization matrix extraction, blocking artifact index, DCT histogram comb metrics."
            },
            {
                "id": "c2pa",
                "name": "C2PA Provenance Verification",
                "supports": ["JPEG", "PNG", "WEBP"],
                "description": "JUMBF box verification and cryptographic manifest inspection via c2patool."
            }
        ],
        "guarantee_disclaimer": "This system employs deterministic forensic signal processing. It does NOT use unverified AI classification."
    }
