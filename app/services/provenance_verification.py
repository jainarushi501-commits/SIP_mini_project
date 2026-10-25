import shutil
import subprocess
from typing import Dict, Any, Tuple
from app.schemas import ProvenanceStatus

def verify_c2pa(image_bytes: bytes, original_filename: str) -> Tuple[ProvenanceStatus, Dict[str, Any]]:
    """
    Forensic Algorithm I: C2PA Content Credentials & Cryptographic Provenance Verification.
    Inspects image binary for JUMBF (JPEG Universal Metadata Box Format) boxes
    and delegates to c2patool CLI if installed in the host system.
    """
    citations = [
        "Coalition for Content Provenance and Authenticity (C2PA) Technical Specification v1.3. https://spec.c2pa.org/specifications/"
    ]
    limitations = [
        "A valid C2PA signature proves the origin claim and cryptographic binding to the asset at signing time.",
        "A valid signature does NOT establish that the physical scene depicted is factually true.",
        "Missing credentials do NOT imply that an image is forged; most modern cameras and workflows do not yet sign media."
    ]

    # Quick deterministic binary check for C2PA / JUMBF signatures
    # C2PA embeds in JUMBF boxes with UUID/type 'c2pa' or 'jumb'
    has_c2pa_box = (b"c2pa" in image_bytes) or (b"jumb" in image_bytes) or (b"C2PA" in image_bytes)

    if not has_c2pa_box:
        return ProvenanceStatus.CREDENTIALS_ABSENT, {
            "status": ProvenanceStatus.CREDENTIALS_ABSENT,
            "message": "No C2PA Content Credentials manifest or JUMBF boxes found in image binary.",
            "citations": citations,
            "limitations": limitations
        }

    # If C2PA boxes are detected, check for c2patool CLI
    c2patool_path = shutil.which("c2patool")
    if not c2patool_path:
        return ProvenanceStatus.TRUST_STATUS_UNKNOWN, {
            "status": ProvenanceStatus.TRUST_STATUS_UNKNOWN,
            "message": "C2PA / JUMBF manifest detected in binary, but 'c2patool' is not installed in the host environment to verify cryptographic signatures.",
            "c2pa_box_detected": True,
            "citations": citations,
            "limitations": limitations
        }

    # Run c2patool safely via subprocess
    import tempfile
    from pathlib import Path
    import json

    with tempfile.NamedTemporaryFile(suffix=Path(original_filename).suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        res = subprocess.run(
            [c2patool_path, tmp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if res.returncode == 0:
            parsed = json.loads(res.stdout)
            return ProvenanceStatus.CREDENTIALS_VERIFIED, {
                "status": ProvenanceStatus.CREDENTIALS_VERIFIED,
                "manifest": parsed,
                "citations": citations,
                "limitations": limitations
            }
        else:
            return ProvenanceStatus.VERIFICATION_FAILED, {
                "status": ProvenanceStatus.VERIFICATION_FAILED,
                "error": res.stderr,
                "citations": citations,
                "limitations": limitations
            }
    except Exception as e:
        return ProvenanceStatus.VERIFICATION_FAILED, {
            "status": ProvenanceStatus.VERIFICATION_FAILED,
            "error": str(e),
            "citations": citations,
            "limitations": limitations
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
