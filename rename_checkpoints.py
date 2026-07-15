"""
rename_checkpoints.py

IMPORTANT: RUN THIS SCRIPT WITH --apply ONLY ONCE.
Why: it renames files in place, and running it again can cause
confusion or unintended renames once the codebase has moved to the
new filename format.

One-time migration script. Renames every checkpoint file from the old
naming format to the new naming format:

    Old: {prefix}_model_{stage}_{run}_{epoch}.zip
    New: {prefix}_model_{run}_{stage}_{epoch}.zip

This swaps the order of the stage and run numbers so the run number
comes first, making it easier to track which files belong to the same
training run when looking at a flat list of filenames.

Run it once before switching extract_stage() and the training scripts
over to the new format. After running, verify a few folders by hand
before deleting this script or running it again.

Usage:
    python rename_checkpoints.py          # dry run, only prints what would happen
    python rename_checkpoints.py --apply  # actually renames the files
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"

# Prefixes this script knows how to rename. Add to this list if a new
# checkpoint family is introduced later.
KNOWN_PREFIXES = ["safe_ppo_model", "nodrift_ppo_model"]


def parse_old_filename(filename: str):
    """Parse a checkpoint filename in the old stage_run_epoch format.

    Args:
        filename: Filename without the .zip extension, for example
            "safe_ppo_model_6_12_9".

    Returns:
        A tuple of (prefix, stage, run, epoch) if the filename matches
        a known prefix and has exactly three trailing numbers. Returns
        None if the filename does not match the expected pattern.
    """
    for prefix in KNOWN_PREFIXES:
        if not filename.startswith(prefix + "_"):
            continue

        remainder = filename[len(prefix) + 1:]
        parts = remainder.split("_")

        if len(parts) != 3:
            continue

        try:
            stage, run, epoch = (int(p) for p in parts)
        except ValueError:
            continue

        return prefix, stage, run, epoch

    return None


def main():
    apply_changes = "--apply" in sys.argv

    if not CHECKPOINT_ROOT.exists():
        print(f"No checkpoint root found at {CHECKPOINT_ROOT}")
        return

    total_renamed = 0
    total_skipped = 0

    # Walk every run folder under data/checkpoints.
    for run_folder in sorted(CHECKPOINT_ROOT.iterdir()):
        if not run_folder.is_dir():
            continue

        for checkpoint_file in sorted(run_folder.glob("*.zip")):
            filename = checkpoint_file.stem  # filename without .zip
            parsed = parse_old_filename(filename)

            if parsed is None:
                # Does not match a known checkpoint naming pattern.
                # Likely something like final_model.zip; leave it alone.
                print(f"  Skipping (no match): {checkpoint_file.name}")
                total_skipped += 1
                continue

            prefix, stage, run, epoch = parsed
            new_filename = f"{prefix}_{run}_{stage}_{epoch}.zip"
            new_path = checkpoint_file.parent / new_filename

            if new_path.exists():
                print(f"  SKIP, target already exists: {new_path.name}")
                total_skipped += 1
                continue

            print(f"  {checkpoint_file.name}  ->  {new_filename}")

            if apply_changes:
                checkpoint_file.rename(new_path)

            total_renamed += 1

    print(f"\n{'Renamed' if apply_changes else 'Would rename'}: {total_renamed} files")
    print(f"Skipped: {total_skipped} files")

    if not apply_changes:
        print("\nThis was a dry run. No files were changed.")
        print("Run with --apply to actually rename the files.")


if __name__ == "__main__":
    main()
