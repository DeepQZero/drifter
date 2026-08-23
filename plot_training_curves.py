"""Plot dock rate over training epochs for one or more runs.

The plot shows each run's success-rate curve, including curriculum stages,
threshold progress, stalls, and regressions.

Data sources, checked in this order:
1. run_metadata.json: results.epoch_history for the full curve.
2. epoch_log.csv: standalone-run data saved one row at a time.
3. results.stage_summary: older runs shown as one point per stage.

Runs without any of these sources are reported and skipped. This script only
reads recorded training data; it does not re-run a policy.

Usage:
1. Set RUN_FOLDERS below to the runs to plot.
2. Run: python plot_training_curves.py

Supported folders include safe_PPO_*,
nodrift_curriculum_PPO_*, nodrift_PPO_*, and
nodrift_standalone_PPO_*. These may be mixed on one figure.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from evaluation_utilities import (
    save_and_show_figure,
    plural_word,
)

# --- Settings ---

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = BASE_DIR / "data" / "checkpoints"

# Run folders to plot, by folder name under data/checkpoints.
# One entry plots a single curve; multiple entries overlay on shared axes.
RUN_FOLDERS = [
    "latest",
    "safe_PPO_55",
]

# True:  pop up a window for each plot.
# False: save each plot's PNG to saved_figures/ and skip the window.
# Also skipped automatically under a headless backend (e.g. MPLBACKEND=Agg).
SHOW_PLOTS = True

# True:  shade each curriculum stage's span and label it along the top.
# False: plain curve, no stage shading. Use when comparing runs whose
# stage boundaries differ.
SHADE_STAGES = True

# True: add a second panel plotting mean fuel (Delta-v, L1) per epoch.
# Only nodrift curriculum runs record fuel data; the panel is dropped
# if no plotted run has any.
PLOT_FUEL_PANEL = True

# Okabe-Ito colorblind-safe palette; avoids red/blue (reserved for win/loss).
RUN_COLORS = ["#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#56B4E9"]


def load_epoch_history(run_folder):
    """Read one run's per-epoch training history from its metadata.

    Args:
        run_folder: Folder name under data/checkpoints, e.g. "safe_PPO_39".

    Returns:
        A (rows, source) tuple. rows is a list of dicts each carrying at
        least "stage", "epoch", and "dock_rate"; source is "epoch_history"
        for a real per-epoch curve, "stage_summary" for the coarser
        one-point-per-stage fallback, or None when the run has neither
        (rows is then empty).
    """
    metadata_path = CHECKPOINT_ROOT / run_folder / "run_metadata.json"
    metadata = None
    if metadata_path.exists():
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            # Unreadable metadata falls through to epoch_log.csv below.
            metadata = None

    if metadata is not None:
        results = metadata.get("results") or {}
        history = results.get("epoch_history")
        if history:
            return list(history), "epoch_history"
    else:
        results = {}

    # Fallback when run_metadata.json is missing or unreadable.
    # epoch_log.csv holds the same per-epoch data, flushed one row at a time.
    csv_path = CHECKPOINT_ROOT / run_folder / "epoch_log.csv"
    if csv_path.exists():
        rows = []
        try:
            with open(csv_path, encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    # Skip unparseable rows, such as a truncated last row,
                    # instead of giving up on the whole file. "stage" uses
                    # .get() since older CSVs lack that column.
                    try:
                        parsed = {
                            "stage": int(row["stage"]) if row.get("stage") else None,
                            "epoch": int(row["epoch"]),
                            "timesteps": int(row["timesteps"]),
                            "dock_rate": float(row["dock_rate"]),
                        }
                        # Optional: only the nodrift trainers write fuel_l1.
                        if row.get("fuel_l1"):
                            parsed["fuel_l1"] = float(row["fuel_l1"])
                        rows.append(parsed)
                    except (TypeError, ValueError, KeyError):
                        continue
        except OSError:
            rows = []
        if rows:
            return rows, "epoch_log.csv"

    if metadata is None:
        return [], None

    # Fallback: one point per stage using its best dock rate (upper envelope,
    # not the real curve). x_epoch aligns it to the same x scale as epoch_history.
    stage_summary = results.get("stage_summary")
    if stage_summary:
        rows = []
        cumulative = 0
        for entry in stage_summary:
            epochs_run = entry.get("epochs_run", 1)
            cumulative += epochs_run
            rows.append({
                "stage": entry.get("stage"),
                "epoch": epochs_run - 1,
                "x_epoch": cumulative - 1,
                "epochs_run": epochs_run,
                "dock_rate": entry.get("best_dock_rate"),
            })
        return rows, "stage_summary"

    return [], None


def cumulative_epochs(rows):
    """Give each row its run-wide x position, not its within-stage epoch.

    A curriculum run restarts its epoch counter at 0 every stage, so
    plotting raw "epoch" would stack all stages on top of each other at
    the left of the axes. This gives each row a run-wide x position, so
    later stages sit further right (the "curves shifting right" shape a
    curriculum run should have).

    Rows carrying an explicit "x_epoch" (the stage_summary fallback, one
    point per stage) use it, so those points land at the true run-wide
    epoch each stage ended on rather than at their index in the list.
    Real per-epoch rows are already one-per-epoch in training order, so
    their position is just their index.

    Args:
        rows: Per-epoch rows, in training order.

    Returns:
        A list of run-wide x positions, same length as rows.
    """
    if rows and rows[0].get("x_epoch") is not None:
        return [row["x_epoch"] for row in rows]
    return list(range(len(rows)))


def stage_spans(rows):
    """Find where each curriculum stage starts and ends along the x axis.

    Args:
        rows: Per-epoch rows, in training order.

    Returns:
        A list of (stage, start_x, end_x) tuples, one per stage, in the
        same x coordinates cumulative_epochs() produces, so shading and
        labels line up with the plotted points (inclusive on both ends).
    """
    spans = []
    if not rows:
        return spans

    x = cumulative_epochs(rows)
    current_stage = rows[0].get("stage")
    # A fallback row covers its whole stage, so its span starts
    # epochs_run - 1 before the point, not at the point itself.
    start = x[0] - (rows[0].get("epochs_run", 1) - 1)
    for i, row in enumerate(rows):
        if row.get("stage") != current_stage:
            spans.append((current_stage, start, x[i - 1]))
            current_stage = row.get("stage")
            start = x[i] - (row.get("epochs_run", 1) - 1)
    spans.append((current_stage, start, x[-1]))
    return spans


def print_epoch_table(run_folder, rows, source):
    """Print a stage-by-epoch grid of dock rates for one run.

    The console counterpart to the plot: a table with one row per
    curriculum stage and one column per epoch within that stage, so exact
    values are readable instead of estimated off a curve.

    Args:
        run_folder: The run's folder name, used as the table heading.
        rows: Per-epoch rows for that run.
        source: Which data source the rows came from, noted in the
            heading so a coarse stage_summary fallback is not mistaken
            for a real per-epoch curve.
    """
    print(f"\n{run_folder}  (source: {source})")
    if source == "stage_summary":
        print("  Per-stage best dock rate only; this source carries "
              "one value per stage, not one per epoch.")

    by_stage = {}
    for row in rows:
        by_stage.setdefault(row.get("stage"), []).append(row)

    max_epochs = max((len(v) for v in by_stage.values()), default=0)
    header = "  Stage  " + "".join(f"{i:>8}" for i in range(max_epochs))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stage in sorted(by_stage, key=lambda s: (s is None, s)):
        cells = "".join(
            f"{row['dock_rate']:>8.3f}" if row.get("dock_rate") is not None else f"{'n/a':>8}"
            for row in by_stage[stage]
        )
        print(f"  {stage:<5}  {cells}")


def plot_training_curves(runs):
    """Plot dock rate (and optionally fuel) per epoch for one or more runs.

    Args:
        runs: List of (run_folder, rows, source) tuples, already filtered
            to runs that have data.
    """
    has_fuel = any(
        any(row.get("fuel_l1") is not None for row in rows)
        for _, rows, _ in runs
    )
    draw_fuel = PLOT_FUEL_PANEL and has_fuel

    if draw_fuel:
        fig, (ax_rate, ax_fuel) = plt.subplots(
            2, 1, figsize=(13, 9), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
    else:
        fig, ax_rate = plt.subplots(figsize=(13, 6.5))
        ax_fuel = None

    # Stage shading is drawn from the first run only; multiple runs'
    # stage boundaries would overlap into noise.
    if SHADE_STAGES and runs:
        import matplotlib.transforms as mtransforms
        # x in data coords, y in axes-fraction: places the label above the
        # plot box so it never collides with a curve near dock_rate=1.0.
        label_transform = mtransforms.blended_transform_factory(
            ax_rate.transData, ax_rate.transAxes)

        first_rows = runs[0][1]
        for i, (stage, start, end) in enumerate(stage_spans(first_rows)):
            if i % 2 == 1:
                ax_rate.axvspan(start - 0.5, end + 0.5, color="black", alpha=0.045, zorder=0)
            mid = (start + end) / 2
            ax_rate.text(
                mid, 1.015, f"stage {stage}", transform=label_transform,
                ha="center", va="bottom", fontsize=8, color="#444444",
                clip_on=False,
            )

    for i, (run_folder, rows, source) in enumerate(runs):
        color = RUN_COLORS[i % len(RUN_COLORS)]
        x = cumulative_epochs(rows)
        y = [row.get("dock_rate") for row in rows]

        # A stage_summary fallback is one point per stage, not a real curve,
        # so it's drawn as markers on a dashed line and labeled as such.
        is_coarse = source == "stage_summary"
        if is_coarse:
            # len(rows) is per STAGE here, not per epoch; the real epoch count
            # is the last x position plus one, so report both instead of
            # mislabeling stage points as epochs.
            total_epochs = (x[-1] + 1) if x else 0
            label = (f"{run_folder} ({total_epochs} "
                     f"{plural_word(total_epochs, 'epoch')} over {len(rows)} "
                     f"{plural_word(len(rows), 'stage')}), per-stage best only")
        else:
            label = f"{run_folder} ({len(rows)} {plural_word(len(rows), 'epoch')})"

        ax_rate.plot(
            x, y,
            color=color,
            linewidth=1.8,
            linestyle="--" if is_coarse else "-",
            marker="o" if is_coarse else None,
            markersize=5,
            alpha=0.9,
            label=label,
            zorder=3,
        )

        # Mark each stage's last epoch, where it cleared its threshold.
        # Walks rows directly rather than slicing by stage_spans' x coordinates,
        # since those are plot positions, not list indices.
        if not is_coarse:
            stage_end_x, stage_end_y = [], []
            for j, row in enumerate(rows):
                is_last_of_stage = (
                    j == len(rows) - 1
                    or rows[j + 1].get("stage") != row.get("stage")
                )
                if is_last_of_stage and row.get("dock_rate") is not None:
                    stage_end_x.append(x[j])
                    stage_end_y.append(row["dock_rate"])
            ax_rate.scatter(
                stage_end_x, stage_end_y,
                color=color, s=28, zorder=4, edgecolors="black", linewidths=0.5,
            )

        if ax_fuel is not None:
            fuel_x = [xi for xi, row in zip(x, rows) if row.get("fuel_l1") is not None]
            fuel_y = [row["fuel_l1"] for row in rows if row.get("fuel_l1") is not None]
            if fuel_y:
                ax_fuel.plot(fuel_x, fuel_y, color=color, linewidth=1.6,
                             alpha=0.9, label=run_folder, zorder=3)

    ax_rate.set_ylabel("Dock rate (success rate)")
    ax_rate.set_ylim(-0.03, 1.03)
    ax_rate.grid(alpha=0.3)
    ax_rate.legend(loc="lower right", fontsize=9)
    title = "Training progress: dock rate per epoch"
    if len(runs) == 1:
        title += f"\n{runs[0][0]}"
    else:
        title += f"\n{len(runs)} runs compared"
    # Extra pad: stage-shading labels sit just above the axes box
    # (y=1.015) and would crowd the title without it.
    ax_rate.set_title(title, fontsize=12, pad=24 if SHADE_STAGES else 6)

    if ax_fuel is not None:
        ax_fuel.set_ylabel("Delta-v, per-axis sum\n(L1, m/s)")
        ax_fuel.grid(alpha=0.3)
        ax_fuel.legend(loc="upper right", fontsize=9)
        ax_fuel.set_xlabel("Epoch (cumulative across the whole run)")
    else:
        ax_rate.set_xlabel("Epoch (cumulative across the whole run)")

    plt.tight_layout()
    save_and_show_figure(BASE_DIR, "training_curves", "training_curves",
                         "training curve plot", show=SHOW_PLOTS)


if __name__ == "__main__":
    print(f"Checkpoint root: {CHECKPOINT_ROOT}")

    loaded = []
    for run_folder in RUN_FOLDERS:
        rows, source = load_epoch_history(run_folder)
        if not rows:
            print(f"\n{run_folder}: no epoch_history or stage_summary in "
                  f"run_metadata.json, skipping. Older runs may not have "
                  f"training history; retrain, or plot a newer run.")
            continue
        loaded.append((run_folder, rows, source))
        print_epoch_table(run_folder, rows, source)

    if not loaded:
        raise SystemExit(
            "\nNo run in RUN_FOLDERS had training history to plot. Set "
            "RUN_FOLDERS to a run that recorded per-epoch history."
        )

    plot_training_curves(loaded)
