"""
rename_checkpoints.py

Migration script that renames every checkpoint under data/checkpoints to
this project's canonical format: {prefix}_model_{run}_{stage}_{epoch}.zip

Older formats it handles (see evaluation_utilities.py for the full history):
    - {prefix}_model_{stage}_{run}_{epoch}.zip  (stage/run swapped)
    - {prefix}_model_{stage}_{run}.zip          (earliest, no epoch yet)

Reads each file's (run, stage, epoch) via evaluation_utilities.
_parse_checkpoint rather than re-guessing the format itself. That
function clarifies the current vs. swapped 3-number formats by
cross-checking the run number against the checkpoint's own parent
folder name (example: "safe_PPO_15" -> run 15).

This makes the script safe to run on a mix of migrated and not-yet-migrated 
folders: a file already in canonical format parses back to the same 
(run, stage, epoch), so its target filename matches its current one and it 
is left alone.

Usage:
    python rename_checkpoints.py          # Dry run, only prints what would happen
    python rename_checkpoints.py --apply  # Actually renames the files
"""

import os
import sys
from pathlib import Path

from evaluation_utilities import _parse_checkpoint

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"

# Prefixes this script knows how to rename. Add to this list if a new
# checkpoint family is introduced later.
KNOWN_PREFIXES = ["safe_ppo_model", "nodrift_ppo_model"]


def main():
    apply_changes = "--apply" in sys.argv

    if not CHECKPOINT_ROOT.exists():
        print(f"No checkpoint root found at {CHECKPOINT_ROOT}")
        return

    total_renamed = 0
    total_unchanged = 0
    total_skipped = 0

    # Walk every run folder under data/checkpoints.
    for run_folder in sorted(CHECKPOINT_ROOT.iterdir()):
        if not run_folder.is_dir():
            continue

        for checkpoint_file in sorted(run_folder.glob("*.zip")):
            filename = checkpoint_file.stem  # filename without .zip

            prefix = next(
                (p for p in KNOWN_PREFIXES if filename.startswith(p + "_")), None
            )
            if prefix is None:
                # Does not match a known checkpoint naming pattern.
                # Likely something like final_model.zip; leave it alone.
                print(f"  Skipping (no match): {checkpoint_file.name}")
                total_skipped += 1
                continue

            run, stage, epoch = _parse_checkpoint(str(checkpoint_file))
            if -1 in (run, stage, epoch):
                # Prefix matched but the trailing numbers didn't parse
                # into a complete (run, stage, epoch). Leave it alone
                # rather than guessing.
                print(f"  Skipping (could not parse): {checkpoint_file.name}")
                total_skipped += 1
                continue

            new_filename = f"{prefix}_{run}_{stage}_{epoch}.zip"
            new_path = checkpoint_file.parent / new_filename

            if new_path == checkpoint_file:
                # Already in the canonical format. This is what makes
                # the script idempotent: nothing to rename here.
                total_unchanged += 1
                continue

            if new_path.exists():
                print(f"  SKIP, target already exists: {new_path.name}")
                total_skipped += 1
                continue

            print(f"  {checkpoint_file.name}  ->  {new_filename}")

            if apply_changes:
                checkpoint_file.rename(new_path)

            total_renamed += 1

    print(f"\n{'Renamed' if apply_changes else 'Would rename'}: {total_renamed} files")
    print(f"Already correct: {total_unchanged} files")
    print(f"Skipped: {total_skipped} files")

    if not apply_changes:
        print("\nThis was a dry run. No files were changed.")
        print("Run with --apply to actually rename the files.")


if __name__ == "__main__":
    main()
