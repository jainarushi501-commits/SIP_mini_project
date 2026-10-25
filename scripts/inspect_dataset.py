#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.dataset_loader import inspect_cg1050_dataset

def main():
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.CG1050_DIR
    print(f"[*] Scanning dataset in: {target_dir}")
    manifest, summary = inspect_cg1050_dataset(target_dir)

    print("\n================ DATASET INSPECTION SUMMARY ================")
    print(f"Directory Exists:      {summary['exists']}")
    print(f"Total Files Scanned:   {summary['total_files_scanned']}")
    print(f"Valid Authentic/Manip: {summary['valid_pairs_count']} pairs")
    print(f"Invalid / Unmatched:   {summary['invalid_pairs_count']}")
    print(f"Unique Source Images:  {summary['unique_source_images']}")
    print("Detected Types:       ", summary['types_detected'])
    
    if summary['errors']:
        print("\nValidation Errors / Warnings:")
        for err in summary['errors']:
            print(f"  - {err}")

    if manifest:
        print("\nSample Valid Pairs:")
        for item in manifest[:3]:
            print(f"  [{item.pair_id}] Source:{item.source_id} | Type:{item.manipulation_type}")
            print(f"     Auth:  {item.authentic_path}")
            print(f"     Manip: {item.manipulated_path}")

if __name__ == "__main__":
    main()
