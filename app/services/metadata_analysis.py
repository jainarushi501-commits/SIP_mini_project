from typing import Dict, Any, List
from PIL import Image, ExifTags
from app.schemas import ModuleResult, ModuleStatus

def analyze_metadata(img: Image.Image, raw_bytes: bytes, original_format: str) -> ModuleResult:
    """
    Forensic Algorithm A: Metadata and File Characteristics Inspection.
    Extracts EXIF, inspects software tags, timestamps, camera details, and format characteristics.
    """
    citations = [
        "ISO 12234-2: Electronic still-picture imaging — Removable memory — Part 2: TIFF/EP image data format",
        "Kee, E., Johnson, M. K., & Farid, H. (2011). Digital image authentication from JPEG features."
    ]
    limitations = [
        "Metadata can be stripped, copied, or modified without altering image pixels.",
        "Absence of EXIF metadata is common in web images and social media, not proof of tampering.",
        "Integrity hash identifies uploaded byte payload, not whether the physical scene was real."
    ]
    confounding_factors = [
        "Messaging apps (WhatsApp, Telegram) and platforms (X, Instagram) routinely strip all EXIF tags.",
        "Privacy software deliberately scrubs GPS and camera serial tags."
    ]

    extracted_exif: Dict[str, Any] = {}
    suspicious_indicators: List[str] = []
    contradictions: List[str] = []

    # Safe extraction from PIL
    try:
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Filter out raw binary blobs for clean JSON serialization
                if isinstance(value, bytes):
                    extracted_exif[tag_name] = f"<binary data {len(value)} bytes>"
                else:
                    extracted_exif[tag_name] = str(value)

            # Check IFD (Exif sub-IFD) for camera settings & timestamps
            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                try:
                    ifd = raw_exif.get_ifd(ifd_id)
                    for tag_id, value in ifd.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if isinstance(value, bytes):
                            extracted_exif[tag_name] = f"<binary data {len(value)} bytes>"
                        else:
                            extracted_exif[tag_name] = str(value)
                except Exception:
                    pass
    except Exception as e:
        extracted_exif["_exif_read_error"] = str(e)

    # Inspect for editing software signatures
    software = extracted_exif.get("Software", "")
    known_editors = ["Photoshop", "GIMP", "Canva", "Lightroom", "Snapseed", "Affinity", "Pixelmator"]
    for ed in known_editors:
        if ed.lower() in software.lower():
            suspicious_indicators.append(f"Image contains editing software signature in EXIF: '{software}'")

    # Timestamp comparison if multiple timestamps exist
    dt_orig = extracted_exif.get("DateTimeOriginal")
    dt_mod = extracted_exif.get("DateTime")
    dt_digitized = extracted_exif.get("DateTimeDigitized")
    if dt_orig and dt_mod and dt_orig != dt_mod:
        contradictions.append(f"Modification timestamp ({dt_mod}) differs from capture timestamp ({dt_orig}).")

    # Determine status and contribution
    measurements = {
        "has_exif": len(extracted_exif) > 0,
        "total_tags_found": len(extracted_exif),
        "camera_make": extracted_exif.get("Make", "Unknown"),
        "camera_model": extracted_exif.get("Model", "Unknown"),
        "software": software or "None recorded",
        "date_original": dt_orig or "None",
        "date_modified": dt_mod or "None",
        "suspicious_indicators": suspicious_indicators,
        "contradictions": contradictions,
        "raw_tags": extracted_exif
    }

    if suspicious_indicators or contradictions:
        status = ModuleStatus.COMPLETED
        contribution = "SUSPICIOUS_METADATA"
    elif len(extracted_exif) == 0:
        status = ModuleStatus.COMPLETED
        contribution = "NO_METADATA_PRESENT"
    else:
        status = ModuleStatus.COMPLETED
        contribution = "CONSISTENT_METADATA"

    return ModuleResult(
        module_name="Metadata & File Inspection",
        status=status,
        verdict_contribution=contribution,
        measurements=measurements,
        thresholds={"known_editing_signatures": known_editors},
        candidate_regions=[],
        visualizations={},
        limitations=limitations,
        confounding_factors=confounding_factors,
        citations=citations
    )
