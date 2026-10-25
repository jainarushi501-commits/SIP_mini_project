import os
import tempfile
from pathlib import Path
from pydantic_settings import BaseSettings

# Determine a guaranteed writable directory for runtime artifacts.
# Operating system tempdir (e.g. /tmp on Linux/Vercel/Lambda, /tmp or /var/folders on macOS)
# is ALWAYS writable across all platforms, including serverless environments.
_SYSTEM_TMP_DIR = Path(tempfile.gettempdir()) / "forensight"

class Settings(BaseSettings):
    APP_NAME: str = "ForenSight"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STATIC_DIR: Path = BASE_DIR / "static"
    
    # Writable runtime directories:
    # Always default to the operating system's writable tempdir (_SYSTEM_TMP_DIR)
    # unless explicitly overridden via environment variable.
    # This completely eliminates any risk of writing to read-only paths (such as /var/task).
    GENERATED_DIR: Path = Path(os.environ.get("FORENSIGHT_GENERATED_DIR") or (_SYSTEM_TMP_DIR / "generated"))
    REPORTS_DIR: Path = Path(os.environ.get("FORENSIGHT_REPORTS_DIR") or (_SYSTEM_TMP_DIR / "reports"))
    
    DATA_DIR: Path = BASE_DIR / "data"
    CG1050_DIR: Path = Path(os.environ.get("FORENSIGHT_DATASET_DIR") or (DATA_DIR / "CG-1050"))
    
    # Upload and security limits
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
    MAX_IMAGE_PIXELS: int = 25_000_000  # 25 MP decomp limit
    MAX_DIMENSION: int = 8000
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
    ALLOWED_MIME_TYPES: set = {"image/jpeg", "image/png", "image/webp"}
    
    # Forensic module parameters
    ELA_QUALITY: int = 90
    ELA_SCALE: int = 15
    FFT_MAX_PEAKS: int = 20
    SIFT_MAX_FEATURES: int = 2000
    SIFT_RATIO_THRESH: float = 0.75
    MIN_COPY_MOVE_MATCHES: int = 10
    MIN_KEYPOINT_DISTANCE: float = 30.0  # Min pixel distance to reject self/trivial neighbor matches

    model_config = {
        "env_file": ".env",
        "env_prefix": "FORENSIGHT_",
        "extra": "ignore"
    }

settings = Settings()

def ensure_runtime_dirs() -> None:
    """Safely creates writable runtime directories without failing if read-only."""
    for d in (settings.GENERATED_DIR, settings.REPORTS_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # Optional dataset dir: only create if parent is writable
    try:
        settings.CG1050_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# Ensure directories exist safely
ensure_runtime_dirs()
