"""
Dataset Validation Script — Sentry Vision AI Security System
==============================================================
Validates dataset integrity for CI pipeline. Checks:
  - Required directory structure exists
  - Image files are readable and non-corrupt
  - Video files are accessible
  - Label distribution is reasonable (not all one class)

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --data-dir data/fire_dataset
"""

import sys
import os
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate dataset integrity")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to dataset root (default: from config)"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on any warning (e.g., minor imbalance)"
    )
    return parser.parse_args()


def check_directory_structure(data_dir):
    """Verify that the dataset directory exists and is not empty."""
    errors = []

    if not os.path.exists(data_dir):
        errors.append(f"Dataset directory does not exist: {data_dir}")
        return errors

    if not os.path.isdir(data_dir):
        errors.append(f"Dataset path is not a directory: {data_dir}")
        return errors

    # Count files recursively
    total_files = 0
    for _, _, files in os.walk(data_dir):
        total_files += len(files)

    if total_files == 0:
        errors.append(f"Dataset directory is empty: {data_dir}")
    else:
        print(f"  ✓ Dataset directory exists with {total_files} file(s)")

    return errors


def check_image_integrity(data_dir):
    """Verify that image files are readable."""
    errors = []
    warnings = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    checked = 0
    corrupt = 0

    try:
        from PIL import Image
    except ImportError:
        warnings.append("Pillow not installed — skipping image integrity check")
        return errors, warnings

    for root, _, files in os.walk(data_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in image_exts:
                continue
            checked += 1
            filepath = os.path.join(root, f)
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except Exception as e:
                corrupt += 1
                if corrupt <= 10:  # Limit reported errors
                    errors.append(f"Corrupt image: {filepath} ({e})")

    if checked == 0:
        warnings.append("No image files found in dataset")
    else:
        print(f"  ✓ Checked {checked} image(s), {corrupt} corrupt")

    return errors, warnings


def check_video_accessibility(data_dir):
    """Verify that video files exist and are accessible."""
    warnings = []
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    checked = 0
    inaccessible = 0

    for root, _, files in os.walk(data_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in video_exts:
                continue
            checked += 1
            filepath = os.path.join(root, f)
            if not os.access(filepath, os.R_OK):
                inaccessible += 1
                if inaccessible <= 10:
                    warnings.append(f"Inaccessible video: {filepath}")

    if checked > 0:
        print(f"  ✓ Found {checked} video(s), {inaccessible} inaccessible")
    else:
        print("  ℹ No video files found (optional)")

    return warnings


def check_label_distribution(data_dir):
    """Check that the dataset has samples from multiple classes."""
    warnings = []

    try:
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=data_dir, transform=None)
    except Exception as e:
        warnings.append(f"Could not load dataset for label check: {e}")
        return warnings

    if len(ds) == 0:
        warnings.append("Dataset has 0 samples — cannot check distribution")
        return warnings

    label_counts = {}
    for sample in ds.samples:
        label = sample[1]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"  ✓ Label distribution: {label_counts}")

    # Check for extreme imbalance (>95% one class)
    total = sum(label_counts.values())
    for label, count in label_counts.items():
        ratio = count / total
        if ratio > 0.95:
            warnings.append(
                f"Severe class imbalance: label {label} has {ratio:.1%} of samples"
            )

    return warnings


def main():
    args = parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        try:
            from src.fire_detection.config import config
            data_dir = config.DATA_DIR
        except ImportError:
            data_dir = os.path.join(PROJECT_ROOT, "data", "fire_dataset")

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    print(f"  Data Dir: {data_dir}")
    print(f"  Strict:   {args.strict}")
    print("=" * 60)

    all_errors = []
    all_warnings = []

    # 1. Directory structure
    print("\n[1/4] Checking directory structure...")
    errors = check_directory_structure(data_dir)
    all_errors.extend(errors)

    # Stop early if directory doesn't exist
    if errors:
        print(f"\n✗ VALIDATION FAILED — {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    # 2. Image integrity
    print("\n[2/4] Checking image integrity...")
    errors, warnings = check_image_integrity(data_dir)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # 3. Video accessibility
    print("\n[3/4] Checking video accessibility...")
    warnings = check_video_accessibility(data_dir)
    all_warnings.extend(warnings)

    # 4. Label distribution
    print("\n[4/4] Checking label distribution...")
    warnings = check_label_distribution(data_dir)
    all_warnings.extend(warnings)

    # Summary
    print("\n" + "=" * 60)
    if all_warnings:
        for w in all_warnings:
            print(f"  ⚠ WARNING: {w}")

    if all_errors:
        print(f"\n✗ VALIDATION FAILED — {len(all_errors)} error(s)")
        for e in all_errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    elif all_warnings and args.strict:
        print(f"\n✗ VALIDATION FAILED (strict mode) — {len(all_warnings)} warning(s)")
        sys.exit(1)
    else:
        print("\n✓ DATASET VALIDATION PASSED")
        print("=" * 60)


if __name__ == "__main__":
    main()
