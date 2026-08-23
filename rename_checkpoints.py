"""Rename checkpoints to the project's standard naming format.

Standard format:
    {prefix}_model_{run}_{stage}_{epoch}.zip

Handled older formats:
- {prefix}_model_{stage}_{run}_{epoch}.zip
- {prefix}_model_{stage}_{run}.zip
- nodrift_ppo_model_*.zip
  - Legacy nodrift files are renamed to either:
    nodrift_curriculum_ppo_model or nodrift_standalone_ppo_model
    based on the run folder name.

Safe to run repeatedly:
- Already-correct files are left alone.
- Mixed old/new folders can be processed in one run.

Usage:
    python rename_checkpoints.py          # Dry run only
    python rename_checkpoints.py --apply  # Rename files
"""

import sys
from pathlib import Path

from evaluation_utilities import _parse_checkpoint

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"

# Prefixes this script recognizes. None is a prefix of another, so
# matching order is irrelevant.
KNOWN_PREFIXES = [
    "safe_ppo_model",
    "nodrift_curriculum_ppo_model",
    "nodrift_standalone_ppo_model",
    "nodrift_ppo_model",  # legacy: shared by both nodrift trainers, see resolve_target_prefix
]

# Legacy prefix shared by both nodrift trainers before they had distinct
# CHECKPOINT_PREFIX values. Its target prefix must be resolved from the
# run folder name; see resolve_target_prefix.
LEGACY_NODRIFT_PREFIX = "nodrift_ppo_model"


def resolve_target_prefix(matched_prefix: str, run_folder_name: str) -> str:
    """Resolve the standard target prefix for a matched checkpoint file.

    Every prefix maps to itself except LEGACY_NODRIFT_PREFIX, which both
    nodrift trainers share and so is resolved from the run folder name:
    "curriculum" in the name means nodrift_curriculum_ppo_model, else
    nodrift_standalone_ppo_model.

    Args:
        matched_prefix: One of KNOWN_PREFIXES, already matched against
            the filename.
        run_folder_name: Basename of the checkpoint's parent folder, used
            only to disambiguate LEGACY_NODRIFT_PREFIX.

    Returns:
        The standard prefix this checkpoint's filename should use.
    """
    if matched_prefix != LEGACY_NODRIFT_PREFIX:
        return matched_prefix
    return (
        "nodrift_curriculum_ppo_model"
        if "curriculum" in run_folder_name
        else "nodrift_standalone_ppo_model"
    )


def main():
    """Rename checkpoint files to the current naming scheme.

    Safe to run more than once: running it again on already-renamed files does nothing.
    """
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

            matched_prefix = next(
                (p for p in KNOWN_PREFIXES if filename.startswith(p + "_")), None
            )
            if matched_prefix is None:
                # Unknown naming pattern (e.g. final_model_nodrift_standalone_13.zip,
                # final_model_nodrift_curriculum_60.zip, final_stage_model_safe_51.zip);
                # leave it alone.
                print(f"  Skipping (no match): {checkpoint_file.name}")
                total_skipped += 1
                continue

            run, stage, epoch = _parse_checkpoint(str(checkpoint_file))
            if -1 in (run, stage, epoch):
                # Prefix matched but the trailing numbers didn't parse into a
                # complete (run, stage, epoch). Leave it alone rather than guessing.
                print(f"  Skipping (could not parse): {checkpoint_file.name}")
                total_skipped += 1
                continue

            target_prefix = resolve_target_prefix(matched_prefix, run_folder.name)
            new_filename = f"{target_prefix}_{run}_{stage}_{epoch}.zip"
            new_path = checkpoint_file.parent / new_filename

            if new_path == checkpoint_file:
                # Already in canonical format (correct prefix and order).
                # Nothing to rename; this is why re-running the script is safe.
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
        print("Run with --apply to rename the files.")


if __name__ == "__main__":
    main()
