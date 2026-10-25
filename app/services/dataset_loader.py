import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from app.config import settings

class DatasetManifestItem:
    def __init__(
        self,
        pair_id: str,
        source_id: str,
        authentic_path: str,
        manipulated_path: str,
        mask_path: Optional[str] = None,
        manipulation_type: str = "unknown",
        width: int = 0,
        height: int = 0,
        is_valid: bool = True,
        validation_error: Optional[str] = None
    ):
        self.pair_id = pair_id
        self.source_id = source_id
        self.authentic_path = authentic_path
        self.manipulated_path = manipulated_path
        self.mask_path = mask_path
        self.manipulation_type = manipulation_type
        self.width = width
        self.height = height
        self.is_valid = is_valid
        self.validation_error = validation_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_id": self.source_id,
            "authentic_path": self.authentic_path,
            "manipulated_path": self.manipulated_path,
            "mask_path": self.mask_path,
            "manipulation_type": self.manipulation_type,
            "dimensions": [self.width, self.height],
            "is_valid": self.is_valid,
            "validation_error": self.validation_error
        }

def inspect_cg1050_dataset(dataset_dir: Optional[Path] = None) -> Tuple[List[DatasetManifestItem], Dict[str, Any]]:
    """
    Scans, parses, and validates the CG-1050 research dataset structure.
    Pairs authentic and manipulated images based on documented ID naming conventions,
    verifies mask existence and dimension alignment.
    """
    target_dir = dataset_dir or settings.CG1050_DIR
    summary = {
        "dataset_path": str(target_dir),
        "exists": target_dir.exists(),
        "total_files_scanned": 0,
        "valid_pairs_count": 0,
        "invalid_pairs_count": 0,
        "unique_source_images": 0,
        "types_detected": {},
        "errors": []
    }

    if not target_dir.exists():
        summary["errors"].append(f"Dataset directory '{target_dir}' does not exist.")
        return [], summary

    all_files = [p for p in target_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    summary["total_files_scanned"] = len(all_files)

    if len(all_files) == 0:
        summary["errors"].append(f"No image files found in '{target_dir}'.")
        return [], summary

    # Group files by candidate base names
    # CG-1050 typically organizes files by:
    # 1. 'orig' / 'authentic' vs 'tampered' / 'forged' / 'modified' folders or naming
    # 2. Or patterns like {ID}_orig.ext vs {ID}_tampered.ext and {ID}_mask.png
    pairs: Dict[str, Dict[str, Path]] = {}

    for f in all_files:
        stem = f.stem.lower()
        parts = [p.lower() for p in f.parts]
        
        # Check folder hierarchy or filename flags
        is_mask = "mask" in stem or "mask" in parts or "gt" in stem
        is_orig = ("orig" in stem or "authentic" in stem or "pristine" in stem or "original" in parts or "authentic" in parts) and not is_mask
        is_tampered = ("tamper" in stem or "forge" in stem or "manip" in stem or "fake" in stem or "tampered" in parts or "manipulated" in parts) and not is_mask

        # Clean source identifier regex (e.g. im_001, 1050_23, etc.)
        match = re.search(r"(\d+)", stem)
        base_id = match.group(1) if match else re.sub(r"(_orig|_tampered|_mask|_authentic|_fake|_gt).*", "", stem)

        if base_id not in pairs:
            pairs[base_id] = {}

        if is_mask:
            pairs[base_id]["mask"] = f
        elif is_orig:
            pairs[base_id]["orig"] = f
        elif is_tampered:
            pairs[base_id]["tampered"] = f
        else:
            # Fallback based on subdirectory parent name
            parent_name = f.parent.name.lower()
            if "orig" in parent_name or "auth" in parent_name:
                pairs[base_id]["orig"] = f
            elif "tamp" in parent_name or "fake" in parent_name:
                pairs[base_id]["tampered"] = f
            elif "mask" in parent_name:
                pairs[base_id]["mask"] = f

    # Build and validate manifest items
    manifest: List[DatasetManifestItem] = []
    unique_sources = set()

    for base_id, item_paths in pairs.items():
        orig_p = item_paths.get("orig")
        tamp_p = item_paths.get("tampered")
        mask_p = item_paths.get("mask")

        if not orig_p or not tamp_p:
            # Unpaired single image
            continue

        unique_sources.add(base_id)
        is_valid = True
        err_msg = None
        w, h = 0, 0

        # Validate images can be decoded and match dimensions
        try:
            with Image.open(orig_p) as o_img:
                ow, oh = o_img.size
            with Image.open(tamp_p) as t_img:
                tw, th = t_img.size

            if (ow, oh) != (tw, th):
                is_valid = False
                err_msg = f"Dimension mismatch: Authentic ({ow}x{oh}) vs Manipulated ({tw}x{th})"
            else:
                w, h = ow, oh

            if mask_p:
                with Image.open(mask_p) as m_img:
                    mw, mh = m_img.size
                if (mw, mh) != (w, h):
                    is_valid = False
                    err_msg = f"Mask dimension mismatch: ({mw}x{mh}) vs Image ({w}x{h})"

        except Exception as e:
            is_valid = False
            err_msg = f"Decode error: {str(e)}"

        # Infer manipulation type from filename or path (copy-move, splicing, removal)
        path_str = f"{orig_p} {tamp_p}".lower()
        if "copymove" in path_str or "copy_move" in path_str or "cm" in path_str:
            manip_type = "copy-move"
        elif "splice" in path_str or "splicing" in path_str:
            manip_type = "splicing"
        elif "remove" in path_str or "inpainting" in path_str:
            manip_type = "removal"
        else:
            manip_type = "general-tampering"

        summary["types_detected"][manip_type] = summary["types_detected"].get(manip_type, 0) + 1

        manifest_item = DatasetManifestItem(
            pair_id=f"CG1050_{base_id}",
            source_id=base_id,
            authentic_path=str(orig_p),
            manipulated_path=str(tamp_p),
            mask_path=str(mask_p) if mask_p else None,
            manipulation_type=manip_type,
            width=w,
            height=h,
            is_valid=is_valid,
            validation_error=err_msg
        )
        manifest.append(manifest_item)
        if is_valid:
            summary["valid_pairs_count"] += 1
        else:
            summary["invalid_pairs_count"] += 1

    summary["unique_source_images"] = len(unique_sources)
    return manifest, summary
