import hashlib
import io
from pathlib import Path
from typing import Tuple, Dict, Any
from PIL import Image, ImageOps
from app.config import settings
from app.schemas import ImageMetadata

class ImageValidationError(Exception):
    pass

MAGIC_BYTES = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"RIFF": "WEBP",  # Will verify "WEBP" at byte offset 8
}

def validate_and_load_image(raw_bytes: bytes, original_filename: str) -> Tuple[Image.Image, ImageMetadata]:
    """
    Validates upload security, file signatures, decompression bounds,
    and returns a clean PIL Image along with verified ImageMetadata.
    """
    # 1. Size check
    file_size = len(raw_bytes)
    if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
        raise ImageValidationError(f"File size ({file_size} bytes) exceeds limit of {settings.MAX_UPLOAD_SIZE_BYTES} bytes.")
    if file_size < 16:
        raise ImageValidationError("File is too small to be a valid image.")

    # 2. Magic byte / signature check
    detected_format = None
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        detected_format = "JPEG"
    elif raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        detected_format = "PNG"
    elif raw_bytes.startswith(b"RIFF") and len(raw_bytes) >= 12 and raw_bytes[8:12] == b"WEBP":
        detected_format = "WEBP"
    else:
        raise ImageValidationError("Invalid file signature. ForenSight supports genuine JPEG, PNG, and WebP images.")

    # 3. Safe decode & decompression bomb prevention
    try:
        # Enforce pillow decompression limit
        Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
        img_buffer = io.BytesIO(raw_bytes)
        img = Image.open(img_buffer)
        
        # Verify image integrity without reading all pixels yet
        img.verify()
        
        # Re-open because verify() closes or leaves image in unusable state
        img_buffer.seek(0)
        img = Image.open(img_buffer)
        img.load()  # Force decode all pixel buffers safely
    except Exception as e:
        raise ImageValidationError(f"Malformed or corrupt image file: {str(e)}")

    # 4. Check dimensions and bounds
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ImageValidationError(f"Invalid dimensions: {width}x{height}")
    if width > settings.MAX_DIMENSION or height > settings.MAX_DIMENSION:
        raise ImageValidationError(f"Image dimension ({width}x{height}) exceeds maximum allowed ({settings.MAX_DIMENSION}px).")
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise ImageValidationError(f"Pixel count ({width * height}) exceeds safety decompression limit.")

    # 5. Extension vs content mismatch check
    ext = Path(original_filename).suffix.lower()
    expected_exts = {
        "JPEG": {".jpg", ".jpeg"},
        "PNG": {".png"},
        "WEBP": {".webp"}
    }
    warnings = []
    if ext not in expected_exts.get(detected_format, set()):
        warnings.append(f"Extension mismatch: filename has '{ext}' but binary content is {detected_format}.")

    # 6. Compute SHA-256 integrity hash
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    metadata = ImageMetadata(
        filename=Path(original_filename).name,
        sha256=sha256_hash,
        file_size_bytes=file_size,
        format=detected_format,
        width=width,
        height=height,
        color_mode=img.mode,
        warnings=warnings
    )

    return img, metadata
