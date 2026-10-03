"""Generate fixed evaluation start states and save them as JSON.

- All agents use the same positions and velocities for fair comparisons.
- Run and commit the generated files as needed.
- Reusing the same seed and bounds produces the same set.
- Change the seed only when creating a new, non-comparable set.

Usage:
- python make_test_set.py: create the default set, standard_100_v2, which is
    the test set used for the paper results (seed 100, 100-150 m, 0.5 m/s per
    axis). The script refuses to overwrite an existing file unless
    --overwrite is given.
- python make_test_set.py --name <new-name> --expect-fingerprint
    737dce4df767: regenerate the paper test set under a new name; fails
    without writing if the generated set differs from the paper test set.
- python make_test_set.py --num-states 200 --name standard_200: create
    200 named states.
"""

import argparse
from pathlib import Path

from evaluation_utilities import (
    sample_start_states, sample_start_states_paper_distribution, save_test_set,
    test_set_fingerprint,
)

BASE_DIR = Path(__file__).resolve().parent

# Store generated evaluation sets in a dedicated folder.
# We keep them outside the git-ignored data/ folder so they can be
# reviewed and versioned in source control.
TEST_SET_DIR = BASE_DIR / "test_sets"

# Use 100 fixed starting states.
NUM_STATES = 100

# Human-friendly name for this test set. It is also used as the JSON
# filename, so keep it short and descriptive.
NAME = "standard_100_v2"

# Random seed used to generate the states. Do not change this unless you
# intentionally want a brand-new evaluation set; older results are not
# directly comparable to a different seed.
SEED = 100

# Sampling bounds for the starting states.
# These ranges are tight enough to stay within the training distribution of
# both agents, while still covering realistic starting distances and speeds.
MIN_POS_BOUND = 100.0
MAX_POS_BOUND = 150.0
MAX_VEL_BOUND = 0.5
# MAX_VEL_BOUND is the velocity limit for each axis; the true speed can be
# higher because the vector magnitude is still bounded by velocity_limit().

def main():
    """Generate a fixed start-state set and write it to disk.

    Raises:
        FileExistsError: If the target file exists and --overwrite was
            not passed, since replacing it would invalidate every result
            already evaluated on the current set.
    """
    parser = argparse.ArgumentParser(
        description="Generate a fixed evaluation start-state set.",
    )
    parser.add_argument("--num-states", type=int, default=NUM_STATES,
                        help="How many start states (default: %(default)s).")
    parser.add_argument("--name", default=NAME,
                        help="Set name and filename stem (default: %(default)s).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Generation seed (default: %(default)s).")
    parser.add_argument("--max-vel-bound", type=float, default=MAX_VEL_BOUND,
                        help="Maximum starting speed per axis in m/s, used by the "
                             "uniform_cube distribution (default: %(default)s).")
    parser.add_argument("--expect-fingerprint", default=None,
                        help="Fail without writing the file if the generated set's "
                             "fingerprint differs from this value.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing set of the same name.")
    parser.add_argument("--distribution", choices=["uniform_cube", "space_controls", "saferl"],
                        default="uniform_cube",
                        help="Method used to sample the start states. "
                             "'uniform_cube' (default) samples positions and velocities "
                             "uniformly from a cube and rejects unsafe starts, with the "
                             "speed per axis limited by --max-vel-bound. "
                             "'space_controls' samples the radius and angle as in "
                             "Space Controls (2024), with speed up to 80%% of the speed "
                             "limit at the sampled position. "
                             "'saferl' samples as in SafeRL (2022), with speed up to "
                             "100%% of that limit.")
    args = parser.parse_args()

    out_path = TEST_SET_DIR / f"{args.name}.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Overwriting it would invalidate "
            f"every result evaluated on the current set, with no warning. "
            f"Pass --overwrite if that is what you want, or choose a new --name."
        )

    if args.distribution == "uniform_cube":
        bounds = {
            "min_pos_bound": MIN_POS_BOUND,
            "max_pos_bound": MAX_POS_BOUND,
            "max_vel_bound": args.max_vel_bound,
        }
        states = sample_start_states(args.num_states, args.seed, MIN_POS_BOUND,
                                    MAX_POS_BOUND, args.max_vel_bound)
        description = ("Fixed evaluation start states shared by every eval "
                       "script, so all agents face identical episodes.")
    else:
        max_vel_fraction = 0.8 if args.distribution == "space_controls" else 1.0
        bounds = {
            "min_pos_bound": MIN_POS_BOUND,
            "max_pos_bound": MAX_POS_BOUND,
            "max_vel_fraction": max_vel_fraction,
            "distribution": args.distribution,
        }
        states = sample_start_states_paper_distribution(
            args.num_states, args.seed, MIN_POS_BOUND, MAX_POS_BOUND,
            max_vel_fraction=max_vel_fraction)
        paper = {"space_controls": "Space Controls (2024)",
                 "saferl": "SafeRL (2022)"}[args.distribution]
        description = (f"Start states sampled with the radius and angle "
                       f"distribution of {paper}, instead of this project's "
                       f"uniform-cube rejection sampling. Use this set to "
                       f"compare against the numbers reported in that paper "
                       f"on start states drawn the same way.")
    if args.expect_fingerprint is not None:
        actual = test_set_fingerprint(states)
        if actual != args.expect_fingerprint:
            raise SystemExit(
                f"Fingerprint mismatch: generated {actual}, expected "
                f"{args.expect_fingerprint}. Nothing was written."
            )
    save_test_set(
        out_path, states, args.name,
        description=description,
        bounds=bounds, seed=args.seed,
    )

    # Print a quick summary so you can check if the generated
    # states look reasonable before using them for evaluation.
    import numpy as np
    dists = [float(np.linalg.norm(s[0:3])) for s in states]
    speeds = [float(np.linalg.norm(s[3:6])) for s in states]
    print(f"  Distance from chief: {min(dists):.1f} to {max(dists):.1f} m "
          f"(mean {np.mean(dists):.1f})")
    print(f"  Starting speed:      {min(speeds):.3f} to {max(speeds):.3f} m/s "
          f"(mean {np.mean(speeds):.3f})")


if __name__ == "__main__":
    main()
