"""Where are the damaging cells? — analysis of the exhaustive k=1 sweep.

The sweep of PLAN.md step 6 evaluated every single-cell flip of every test map.
That gives a well-powered answer to a question the step-3 pilot could only
guess at from a few hundred random draws: **which cells are dangerous?**

Joins the manifest against the step-4 feature archive on the flipped cell, and
reports the damage rate by location, by whether the cell lies on an optimal
route, and by flip direction.

Usage:
    python scripts/analyse_damaging_cells.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.features import WALL_ZONE_LABELS  # noqa: E402

#: Feature columns joined onto each flipped cell.
JOINED = (
    "wall_zone",
    "on_any_optimal_path",
    "on_reference_path",
    "dist_to_obstacle_chebyshev",
    "corridor_span_min",
    "is_free",
)


def load_joined(manifest_path: Path, split: str) -> pd.DataFrame:
    """Load the exhaustive k=1 rows with their flipped cell's features attached.

    Args:
        manifest_path: The results manifest.
        split: Which split's feature archive to join against.

    Returns:
        pd.DataFrame: One row per single-cell alteration, with the joined
        feature columns and a boolean ``damaging``.

    Raises:
        FileNotFoundError: If either input is missing.
    """
    frame = pd.read_parquet(manifest_path)
    frame = frame[(frame["method"] == "exhaustive") & (frame["budget"] == 1)].copy()
    archive_path = config.RESULTS_DIR / f"features_{split}.npz"
    if not archive_path.exists():
        raise FileNotFoundError(f"{archive_path} missing; run `make features`")
    with np.load(archive_path, allow_pickle=False) as handle:
        archive = {key: handle[key] for key in handle.files}

    index = {int(value): position for position, value in enumerate(archive["instance_ids"])}
    frame["cell_row"] = frame["flip_cells"].apply(lambda cells: int(cells[0][0]))
    frame["cell_col"] = frame["flip_cells"].apply(lambda cells: int(cells[0][1]))
    rows = frame["instance_id"].map(index).to_numpy()
    for name in JOINED:
        frame[name] = archive[name][rows, frame["cell_row"].to_numpy(), frame["cell_col"].to_numpy()]
    # "Damaging" means worse than this instance's own baseline, not worse than
    # optimal: the planner is already suboptimal on 17% of maps.
    frame["damaging"] = frame["opt_C1O1"] > frame["opt_C0O0"]
    return frame


def damage_by(frame: pd.DataFrame, column: str, labels: Dict) -> List[Dict[str, object]]:
    """Damage rate grouped by one joined column.

    Args:
        frame: Output of `load_joined`.
        column: Column to group by.
        labels: Mapping from value to a readable label.

    Returns:
        list[dict]: Per group, the count and the damage rate.
    """
    out = []
    for value, label in labels.items():
        group = frame[frame[column] == value]
        if len(group) < 50:
            continue
        out.append(
            {
                "group": label,
                "n": int(len(group)),
                "damage_rate": float(group["damaging"].mean()),
            }
        )
    return out


def main() -> int:
    """Run the analysis and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=config.RESULTS_DIR / "manifest.parquet")
    args = parser.parse_args()

    frame = load_joined(args.manifest, config.canonical_split(args.split))
    report = {
        "n_alterations": int(len(frame)),
        "n_damaging": int(frame["damaging"].sum()),
        "overall_rate": float(frame["damaging"].mean()),
        "by_wall_zone": damage_by(frame, "wall_zone", WALL_ZONE_LABELS),
        "by_on_optimal_path": damage_by(
            frame, "on_any_optimal_path", {1.0: "on some optimal path", 0.0: "off every optimal path"}
        ),
        "by_direction": damage_by(
            frame, "is_free", {1.0: "free -> wall (pillar)", 0.0: "wall -> free (door)"}
        ),
    }
    free = frame[frame["is_free"] == 1]
    crossed = []
    for on_path, path_label in ((1.0, "on path"), (0.0, "off path")):
        for code, zone_label in WALL_ZONE_LABELS.items():
            group = free[(free["on_any_optimal_path"] == on_path) & (free["wall_zone"] == code)]
            if len(group) < 50:
                continue
            crossed.append(
                {
                    "group": f"{path_label}, {zone_label}",
                    "n": int(len(group)),
                    "damage_rate": float(group["damaging"].mean()),
                }
            )
    report["crossed"] = crossed

    (config.RESULTS_DIR / "e2_damaging_cells.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    (config.RESULTS_DIR / "e2_damaging_cells.md").write_text(render_markdown(report) + "\n")

    print(f"{report['n_alterations']:,} alterations, {report['n_damaging']:,} damaging "
          f"({report['overall_rate']:.2%})\n")
    for section in ("by_wall_zone", "by_on_optimal_path", "by_direction", "crossed"):
        print(section)
        for row in report[section]:
            print(f"   {row['group']:28} n={row['n']:6,}  {row['damage_rate']:.2%}")
        print()
    print("wrote e2_damaging_cells.md and .json")
    return 0


def render_markdown(report: Dict[str, object]) -> str:
    """Render the damaging-cell analysis.

    Args:
        report: Assembled results.

    Returns:
        str: Markdown document.
    """
    lines = [
        "# Where the damaging cells are (exhaustive k=1 sweep)",
        "",
        f"All {report['n_alterations']:,} single-cell flips of the 100 test maps, joined "
        f"against the step-4 feature library on the flipped cell. "
        f"{report['n_damaging']:,} of them ({report['overall_rate']:.2%}) make the planner "
        "worse than its own baseline on that map.",
        "",
        "Because this covers every cell rather than a sample, it supersedes the "
        "direction estimates from the step-3 random pilot, which rested on a few "
        "dozen draws.",
        "",
    ]
    for key, title in (
        ("by_direction", "By what the flip does"),
        ("by_on_optimal_path", "By whether the cell lies on an optimal route"),
        ("by_wall_zone", "By distance to the nearest wall"),
        ("crossed", "Free cells, crossed"),
    ):
        lines += [f"## {title}", "", "| group | cells | damage rate |", "|---|---|---|"]
        for row in report[key]:  # type: ignore[index]
            lines.append(f"| {row['group']} | {row['n']:,} | **{row['damage_rate']:.2%}** |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
