#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.dataset_loader import inspect_cg1050_dataset

def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.CG1050_DIR
    manifest, summary = inspect_cg1050_dataset(target_dir)

    print(f"Validating dataset at: {target_dir}")
    print(f"Found {len(manifest)} potential pairs.")

    errors = 0
    for item in manifest:
        if not item.is_valid:
            errors += 1
            print(f"[FAIL] {item.pair_id}: {item.validation_error}")

    if errors == 0 and len(manifest) > 0:
        print(f"[SUCCESS] All {len(manifest)} image pairs validated successfully.")
    elif len(manifest) == 0:
        print("[NOTICE] No dataset pairs found. Follow instructions in data/CG-1050/README.txt to extract CG-1050.")
    else:
        print(f"[WARNING] Completed with {errors} invalid pairs.")

if __name__ == "__main__":
    main()
