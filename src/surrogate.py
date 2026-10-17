"""Surrogate models of the cost map — E1 (PLAN.md step 5).

The question: how much of the learned cost map can be reconstructed from
quantities a person can name? The method: fit a transparent model from the
feature library (`src.features`) to the encoder's output, on the train split,
and score it on the test split.

Three fits, because they answer different questions:

* **geometric-only** — distances to goal and obstacle structure. This is
  hypothesis H1 of PLAN.md: the cost map is geometry in smoothed form.
* **path-only** — the three path indicators. This is the competing
  hypothesis: the cost map marks where the route goes.
* **both** — how much the two families add to each other.

Two model classes, because the linear and non-linear answers differ sharply
here. Ridge regression is the transparent baseline; a histogram gradient
boosting regressor is the check that a near-zero linear fit is not hiding a
strong curved relationship (it was, for goal geometry: R² 0.007 → 0.105).

A note on what R² can and cannot settle. A cost-map difference only matters if
it changes where the search goes, and nothing guarantees that unexplained
variance is behaviourally relevant. `sufficiency_cost_maps` therefore builds a
cost map *out of the surrogate's predictions* so it can be fed to the real
search module: if planning quality survives, the features captured everything
that matters, whatever the R² says.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .features import AUX_NAMES, FEATURE_NAMES

#: Named groups of features. ``geometric`` and ``path`` partition
#: `FEATURE_NAMES`; the others are nested subsets used for the finer
#: attribution of which geometry matters.
FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "goal_to": ("geodesic_to_goal", "euclidean_to_goal", "chebyshev_to_goal"),
    "goal": (
        "geodesic_to_goal",
        "geodesic_from_start",
        "euclidean_to_goal",
        "chebyshev_to_goal",
    ),
    "heuristic": ("chebyshev_to_goal", "euclidean_to_goal"),
    "obstacle": (
        "dist_to_obstacle_chebyshev",
        "dist_to_obstacle_euclidean",
        "obstacle_density_3x3",
        "obstacle_density_5x5",
        "corridor_width_min4",
        "corridor_span_min",
        "free_neighbours",
    ),
    "path": ("on_reference_path", "on_any_optimal_path", "on_solver_path"),
    #: Hypothesis H1 exactly as PLAN.md states it: distance **to the goal** and
    #: proximity to obstacles. Deliberately excludes `geodesic_from_start`.
    "geometric_h1": (
        "geodesic_to_goal",
        "euclidean_to_goal",
        "chebyshev_to_goal",
        "dist_to_obstacle_chebyshev",
        "dist_to_obstacle_euclidean",
        "obstacle_density_3x3",
        "obstacle_density_5x5",
        "corridor_width_min4",
        "corridor_span_min",
        "free_neighbours",
    ),
    #: Every non-path feature, i.e. `geometric_h1` plus `geodesic_from_start`.
    #: **This group is not a clean test of H1.** A cell lies on some optimal
    #: path exactly when `geodesic_from_start + geodesic_to_goal == optimum`,
    #: so a non-linear model given both distances can reconstruct the path
    #: indicator without being given it. The gap between `geometric_h1` and
    #: `geometric` measures that leakage.
    "geometric": tuple(n for n in FEATURE_NAMES if not n.startswith("on_")),
    "all": tuple(FEATURE_NAMES),
}


@dataclass(frozen=True)
class DesignMatrix:
    """Feature matrix, target and bookkeeping for one split.

    Attributes:
        X: ``(n_cells, n_features)`` float64 matrix, one row per free and
            reachable cell of every instance in the split.
        y: ``(n_cells,)`` the encoder's cost at those cells.
        feature_names: Column names of `X`.
        instance_index: ``(n_cells,)`` which instance each row came from, used
            for the within-instance score.
        wall_zone: ``(n_cells,)`` distance-to-wall class per row.
        topology_class: ``(n_cells,)`` free-neighbour class per row.
        split: Split name.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: Tuple[str, ...]
    instance_index: np.ndarray
    wall_zone: np.ndarray
    topology_class: np.ndarray
    split: str

    def subset(self, group: str) -> "DesignMatrix":
        """Return the same rows restricted to one feature group.

        Args:
            group: A key of `FEATURE_GROUPS`.

        Returns:
            DesignMatrix: With only that group's columns.

        Raises:
            KeyError: If the group is unknown.
        """
        names = FEATURE_GROUPS[group]
        cols = [self.feature_names.index(n) for n in names]
        return DesignMatrix(
            X=self.X[:, cols],
            y=self.y,
            feature_names=tuple(names),
            instance_index=self.instance_index,
            wall_zone=self.wall_zone,
            topology_class=self.topology_class,
            split=self.split,
        )


def load_archive(split: str, results_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load a feature archive built by `scripts/build_features.py`.

    Args:
        split: Canonical split name.
        results_dir: Directory holding the archives; defaults to
            `src.config.RESULTS_DIR`.

    Returns:
        dict[str, np.ndarray]: Every array in the archive.

    Raises:
        FileNotFoundError: If the archive is missing.
    """
    base = results_dir or config.RESULTS_DIR
    path = Path(base) / f"features_{config.canonical_split(split)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `make features` (scripts/build_features.py) first"
        )
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def build_design_matrix(split: str, archive: Optional[Dict[str, np.ndarray]] = None) -> DesignMatrix:
    """Flatten a split's archive into rows, keeping only usable cells.

    Only **free and reachable** cells become rows. Obstacles have no meaningful
    geodesic distance, and free cells outside the goal's connected component
    carry ``inf`` — both would poison a regression, and neither is ever
    expanded by the search.

    Args:
        split: Canonical split name.
        archive: Pre-loaded archive; loaded from disk when omitted.

    Returns:
        DesignMatrix: Rows for every usable cell of the split.
    """
    archive = archive if archive is not None else load_archive(split)
    mask = (archive["is_free"] > 0) & (archive["reachable"] > 0)
    X = np.stack([archive[name][mask] for name in FEATURE_NAMES], axis=1).astype(np.float64)
    y = archive["cost_map"][mask].astype(np.float64)
    per_instance = mask.reshape(mask.shape[0], -1).sum(axis=1)
    instance_index = np.repeat(np.arange(mask.shape[0]), per_instance)
    return DesignMatrix(
        X=X,
        y=y,
        feature_names=tuple(FEATURE_NAMES),
        instance_index=instance_index,
        wall_zone=archive["wall_zone"][mask],
        topology_class=archive["topology_class"][mask],
        split=config.canonical_split(split),
    )


def make_model(kind: str = "ridge", seed: int = 0):
    """Build an unfitted surrogate pipeline.

    Args:
        kind: ``"ridge"`` for standardised ridge regression (the transparent
            baseline) or ``"gbm"`` for histogram gradient boosting (the
            non-linearity check).
        seed: Random seed for the boosting model.

    Returns:
        sklearn.pipeline.Pipeline: Scaler plus estimator.

    Raises:
        ValueError: If `kind` is unknown.

    Note:
        Features are standardised in both cases. Trees do not need it, but it
        keeps the two pipelines interchangeable and the ridge coefficients
        comparable across features.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "ridge":
        estimator = Ridge(alpha=1.0)
    elif kind == "gbm":
        estimator = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
    else:
        raise ValueError(f"unknown model kind {kind!r}; expected 'ridge' or 'gbm'")
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def within_instance_r2(
    y: np.ndarray, prediction: np.ndarray, instance_index: np.ndarray
) -> float:
    """R² computed after removing each instance's mean from target and residual.

    Guards against a fit that looks good only because it separates easy maps
    from hard ones. On this data the cost map's spread is almost entirely
    within instances, so this number should track the pooled R² closely; a gap
    would mean the surrogate is explaining the wrong thing.

    Args:
        y: True values.
        prediction: Predicted values.
        instance_index: Instance id per row.

    Returns:
        float: ``1 − SS_res_centred / SS_tot_centred``.
    """
    residual = y - prediction
    centred_residual = np.empty_like(residual)
    centred_target = np.empty_like(y)
    for instance in np.unique(instance_index):
        rows = instance_index == instance
        centred_residual[rows] = residual[rows] - residual[rows].mean()
        centred_target[rows] = y[rows] - y[rows].mean()
    denominator = float((centred_target**2).sum())
    if denominator == 0:
        return float("nan")
    return float(1.0 - (centred_residual**2).sum() / denominator)


@dataclass
class FitResult:
    """One fitted surrogate and its scores.

    Attributes:
        group: Feature group name.
        kind: ``"ridge"`` or ``"gbm"``.
        n_features: Number of columns used.
        r2: Pooled R² on the evaluation split.
        r2_within: Within-instance R² on the evaluation split.
        rmse: Root mean squared error on the evaluation split.
        model: The fitted pipeline.
        coefficients: Standardised ridge coefficients by feature name, or None.
    """

    group: str
    kind: str
    n_features: int
    r2: float
    r2_within: float
    rmse: float
    model: object
    coefficients: Optional[Dict[str, float]] = None


def fit_and_score(
    train: DesignMatrix, test: DesignMatrix, group: str, kind: str = "ridge", seed: int = 0
) -> FitResult:
    """Fit one surrogate on the train split and score it on the test split.

    Args:
        train: Training design matrix (all features).
        test: Evaluation design matrix (all features).
        group: Feature group to use, a key of `FEATURE_GROUPS`.
        kind: Model kind, see `make_model`.
        seed: Seed for the boosting model.

    Returns:
        FitResult: Scores and the fitted pipeline.
    """
    from sklearn.metrics import r2_score

    train_group, test_group = train.subset(group), test.subset(group)
    model = make_model(kind, seed=seed).fit(train_group.X, train_group.y)
    prediction = model.predict(test_group.X)

    coefficients = None
    if kind == "ridge":
        coefficients = dict(zip(train_group.feature_names, model.named_steps["model"].coef_))
    return FitResult(
        group=group,
        kind=kind,
        n_features=train_group.X.shape[1],
        r2=float(r2_score(test_group.y, prediction)),
        r2_within=within_instance_r2(test_group.y, prediction, test_group.instance_index),
        rmse=float(np.sqrt(np.mean((test_group.y - prediction) ** 2))),
        model=model,
        coefficients=coefficients,
    )


def residual_by_class(
    result: FitResult, test: DesignMatrix, grouping: str = "wall_zone"
) -> List[Dict[str, float]]:
    """Residual magnitude grouped by a cell partition.

    PLAN.md step 5 asks for residuals grouped by local topology. On this map
    family that partition is degenerate (~70 % of cells in one class), so
    ``wall_zone`` — distance to the nearest wall — is the default here and
    ``topology_class`` is available for comparability.

    Args:
        result: A fitted surrogate.
        test: The evaluation design matrix (all features).
        grouping: ``"wall_zone"`` or ``"topology_class"``.

    Returns:
        list[dict]: Per class, the share of cells, mean absolute residual, mean
        signed residual (does the surrogate over- or under-predict there?) and
        the mean target value.

    Raises:
        ValueError: If `grouping` is unknown.
    """
    from .features import TOPOLOGY_LABELS, WALL_ZONE_LABELS

    if grouping == "wall_zone":
        codes, labels = test.wall_zone, WALL_ZONE_LABELS
    elif grouping == "topology_class":
        codes, labels = test.topology_class, TOPOLOGY_LABELS
    else:
        raise ValueError(f"unknown grouping {grouping!r}")

    subset = test.subset(result.group)
    residual = subset.y - result.model.predict(subset.X)
    out = []
    for code in sorted(set(int(c) for c in np.unique(codes))):
        rows = codes == code
        if not rows.any():
            continue
        out.append(
            {
                "class": labels.get(code, str(code)),
                "share": float(rows.mean()),
                "mean_abs_residual": float(np.abs(residual[rows]).mean()),
                "mean_signed_residual": float(residual[rows].mean()),
                "mean_cost": float(subset.y[rows].mean()),
            }
        )
    return out


def sufficiency_cost_maps(
    result: FitResult,
    archive: Dict[str, np.ndarray],
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Turn surrogate predictions back into cost maps the search can run on.

    This is the test that R² cannot perform. The surrogate's prediction is
    assembled into a full ``(N, H, W)`` grid and handed to the same
    differentiable A\\* module the encoder feeds, so planning quality can be
    measured with a cost map built only from interpretable quantities.

    Args:
        result: A fitted surrogate.
        archive: The feature archive for the same split.
        fill_value: Value for cells the surrogate has no row for (obstacles,
            and free cells outside the goal's component). Defaults to the mean
            prediction. These cells are never expanded by the search — it
            cannot leave the start's connected component — so the choice is
            inert; it exists to keep the array rectangular.

    Returns:
        np.ndarray: ``(N, H, W)`` float32 cost maps, clipped to ``(0, 1)`` to
        match the sigmoid range the search was trained against.
    """
    mask = (archive["is_free"] > 0) & (archive["reachable"] > 0)
    names = FEATURE_GROUPS[result.group]
    X = np.stack([archive[name][mask] for name in names], axis=1).astype(np.float64)
    prediction = np.clip(result.model.predict(X), 1e-6, 1.0 - 1e-6)

    filled = float(prediction.mean()) if fill_value is None else float(fill_value)
    out = np.full(archive["cost_map"].shape, filled, dtype=np.float32)
    out[mask] = prediction.astype(np.float32)
    return out


def summarize_fits(results: Sequence[FitResult]) -> List[Dict[str, object]]:
    """Flatten fit results into rows for a report table.

    Args:
        results: Fitted surrogates.

    Returns:
        list[dict]: One row per fit with group, kind, feature count and scores.
    """
    return [
        {
            "group": r.group,
            "kind": r.kind,
            "n_features": r.n_features,
            "r2": r.r2,
            "r2_within": r.r2_within,
            "rmse": r.rmse,
        }
        for r in results
    ]
