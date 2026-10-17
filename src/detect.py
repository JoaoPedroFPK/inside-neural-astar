"""Detecting altered maps from the planner's own internals — E4 (PLAN.md step 12).

The question: given a map the planner has been handed, can we tell that a few
cells were flipped adversarially, without being told? This matters because the
alterations studied in steps 6 and 9 are invisible in the output — the planner
returns a path, it is simply a worse one.

Two detectors, as the plan specifies.

**Residual detector (ours).** Step 5 built a surrogate that predicts the cost
map from interpretable features. Where the encoder behaves as the features
predict, the residual is small. The hypothesis is that an adversarial flip
pushes the encoder somewhere unusual, leaving a residual that is large, or
oddly shaped, or concentrated in one place. Summary statistics of the residual
become the detector's features.

**Mahalanobis baseline.** The standard out-of-distribution score: pool the
encoder's internal activations into a vector per map, estimate a mean and
covariance on clean maps only, and score a new map by its Mahalanobis distance
from that distribution. It is the obvious thing to try and a fair comparator —
if it wins, the surrogate adds nothing.

**What "detection" means here, stated carefully.** The task is a *paired*
comparison: for each instance the detector sees the clean map and an altered
one and must separate them. That is easier than detecting alteration in the
wild, where there is no clean reference for the same map, and the numbers
should be read with that in mind. A detector that cannot do the easy version
would certainly fail the hard one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import MazeInstance
from .grid import RC, flip_cells

#: Order of the residual summary statistics in the feature vector.
RESIDUAL_FEATURES: Tuple[str, ...] = (
    "mean_abs",
    "std",
    "max_abs",
    "p99_abs",
    "frac_over_2sd",
    "max_window_share",
    "signed_mean",
)


def residual_statistics(
    residual: np.ndarray, mask: np.ndarray, window: int = 11, clean_sd: Optional[float] = None
) -> Dict[str, float]:
    """Summarise a cost-map residual into the detector's features.

    Args:
        residual: ``(H, W)`` observed cost map minus the surrogate's prediction.
        mask: ``(H, W)`` boolean, the cells to consider — free and reachable.
        window: Side of the square window used for the concentration measure.
            The default 11 matches the encoder's receptive field, so a single
            flip's whole response fits in exactly one window.
        clean_sd: Residual standard deviation measured on clean maps, used for
            the "fraction of cells beyond two standard deviations" feature.
            Falls back to this map's own spread when omitted, which makes that
            feature scale-free and much weaker.

    Returns:
        dict[str, float]: One entry per name in `RESIDUAL_FEATURES`.
    """
    values = residual[mask]
    if values.size == 0:
        return {name: 0.0 for name in RESIDUAL_FEATURES}
    absolute = np.abs(values)
    spread = clean_sd if clean_sd else float(values.std()) or 1.0

    # Concentration: the largest share of total absolute residual falling in
    # any single window. A diffuse residual spreads it; a localised response
    # piles it into one place.
    padded = np.where(mask, np.abs(residual), 0.0)
    cumulative = padded.cumsum(axis=0).cumsum(axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0)))
    total = float(padded.sum()) or 1.0
    best = 0.0
    height, width = residual.shape
    for row in range(0, max(height - window + 1, 1)):
        for column in range(0, max(width - window + 1, 1)):
            bottom, right = min(row + window, height), min(column + window, width)
            block = (
                cumulative[bottom, right]
                - cumulative[row, right]
                - cumulative[bottom, column]
                + cumulative[row, column]
            )
            best = max(best, float(block))
    return {
        "mean_abs": float(absolute.mean()),
        "std": float(values.std()),
        "max_abs": float(absolute.max()),
        "p99_abs": float(np.percentile(absolute, 99)),
        "frac_over_2sd": float((absolute > 2 * spread).mean()),
        "max_window_share": best / total,
        "signed_mean": float(values.mean()),
    }


#: Encoder layers offered to the activation baseline, and how each is pooled.
#: Four ReLU outputs across the stack, each pooled two ways. The baseline is
#: scored at all eight and reported at its best, so the comparison against the
#: residual detector — which gets one configuration — is conservative.
ACTIVATION_LAYERS: Tuple[int, ...] = (2, 5, 8, 11)


#: Order of the model-free map statistics in the control's feature vector.
MAP_ONLY_FEATURES: Tuple[str, ...] = (
    "obstacle_fraction",
    "isolated_obstacles",
    "isolated_free_cells",
    "dead_ends",
    "optimum_length",
)


def map_only_features(instance: MazeInstance, map_design: np.ndarray) -> Dict[str, float]:
    """Statistics of the map alone — the control that needs no model at all.

    This is the baseline that decides whether the whole exercise is worth
    anything. Flipping a free cell to an obstacle in open ground leaves a
    single obstacle surrounded by free space, which is rare in a generated
    maze and visible to anyone who looks at the grid. If these five numbers
    detect the alteration as well as the planner's internals do, then the
    internals contributed nothing and the honest conclusion is to say so.

    Args:
        instance: Supplies the start and goal for the path-length statistic.
        map_design: ``(H, W)`` map, ``1`` = free.

    Returns:
        dict[str, float]: One entry per name in `MAP_ONLY_FEATURES`.
    """
    from scipy.ndimage import convolve

    from .exact_solver import distance_field

    free = map_design > 0
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    free_neighbours = convolve(free.astype(np.float64), kernel, mode="constant", cval=0.0)
    field = distance_field(map_design, instance.goal_rc)
    return {
        "obstacle_fraction": float((~free).mean()),
        # An obstacle with eight free neighbours: the footprint of a flip.
        "isolated_obstacles": float(((~free) & (free_neighbours == 8)).sum()),
        "isolated_free_cells": float((free & (free_neighbours == 0)).sum()),
        "dead_ends": float((free & (free_neighbours <= 1)).sum()),
        "optimum_length": float(np.nan_to_num(field[instance.start_rc], posinf=-1.0)),
    }


def pooled_activations(
    planner, instance: MazeInstance, map_design: np.ndarray, layers: Sequence[int] = ACTIVATION_LAYERS
) -> Dict[str, np.ndarray]:
    """Pool several encoder layers into feature vectors for the OOD baseline.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: Supplies the start and goal.
        map_design: The obstacle map to encode, altered or not.
        layers: Indices into `planner.encoder.model`. The defaults are the four
            ReLU outputs.

    Returns:
        dict[str, np.ndarray]: Keys ``"l{layer}_mean"`` and ``"l{layer}_max"``,
        each a vector with one entry per channel. The mean asks whether the
        layer's overall activity changed; the max asks whether any one place in
        the map lit up unusually, which is the shape a single flipped cell
        would produce. Pooling discards *where*, which is what makes vectors
        comparable across maps with different geometry.
    """
    from dataclasses import replace

    from .attribution import encoder_input

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(index: int):
        """Build a hook that stores one layer's output."""

        def hook(_module, _inputs, output):
            """Store the layer's output for pooling."""
            captured[index] = output.detach()

        return hook

    for index in layers:
        handles.append(planner.encoder.model[index].register_forward_hook(make_hook(index)))
    try:
        with torch.no_grad():
            planner.encoder(encoder_input(replace(instance, map_design=map_design)))
    finally:
        for handle in handles:
            handle.remove()

    pooled: Dict[str, np.ndarray] = {}
    for index, activation in captured.items():
        pooled[f"l{index}_mean"] = activation.mean(dim=(2, 3))[0].numpy()
        pooled[f"l{index}_max"] = activation.amax(dim=(2, 3))[0].numpy()
    return pooled


@dataclass
class MahalanobisScorer:
    """Distance from the distribution of clean-map activations.

    Attributes:
        mean: Mean activation vector over the clean maps it was fitted on.
        precision: Inverse covariance, regularised so it stays invertible when
            channels are correlated (they are).
    """

    mean: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(cls, vectors: Sequence[np.ndarray], shrinkage: float = 1e-2) -> "MahalanobisScorer":
        """Estimate the clean distribution.

        Args:
            vectors: Activation vectors from clean maps only. Fitting on
                altered maps would leak the thing being detected.
            shrinkage: Ridge added to the covariance diagonal.

        Returns:
            MahalanobisScorer: The fitted scorer.

        Raises:
            ValueError: If fewer than two vectors are supplied.
        """
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) < 2:
            raise ValueError("need at least two clean activation vectors")
        mean = matrix.mean(axis=0)
        covariance = np.cov(matrix, rowvar=False)
        covariance = covariance + shrinkage * np.eye(len(mean)) * np.trace(covariance) / len(mean)
        return cls(mean=mean, precision=np.linalg.inv(covariance))

    def score(self, vector: np.ndarray) -> float:
        """Squared Mahalanobis distance; larger means more unusual.

        Args:
            vector: An activation vector.

        Returns:
            float: The squared distance from the fitted clean distribution.
        """
        delta = np.asarray(vector, dtype=np.float64) - self.mean
        return float(delta @ self.precision @ delta)


def build_examples(
    planner,
    surrogate,
    instances: Sequence[MazeInstance],
    alterations: Dict[int, Sequence[RC]],
    feature_builder,
    clean_sd: Optional[float] = None,
    layers: Sequence[int] = ACTIVATION_LAYERS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, np.ndarray, List[int]]:
    """Build the detector's dataset: one clean and one altered example per instance.

    Args:
        planner: The planner whose encoder is being monitored.
        surrogate: A fitted surrogate (`src.surrogate.FitResult`).
        instances: Instances to use.
        alterations: Instance id -> cells flipped, for the altered example.
        feature_builder: Callable taking ``(instance, map_design)`` and
            returning the surrogate's feature matrix and the usable-cell mask.
        clean_sd: Residual spread measured on clean maps, passed through to
            `residual_statistics`.
        layers: Encoder layers offered to the activation baseline.

    Returns:
        tuple: ``(residual_features, pooled_activations, map_only, labels,
        instance_ids)`` where `pooled_activations` maps each pooling variant to
        its ``(n_examples, n_channels)`` matrix and labels are 1 for altered and
        0 for clean. Rows of every array line up: index ``i`` is the same
        example everywhere.
    """
    from dataclasses import replace

    from .planner import encode_cost_map

    residual_rows: List[List[float]] = []
    activation_rows: Dict[str, List[np.ndarray]] = {}
    map_only_rows: List[List[float]] = []
    labels: List[int] = []
    ids: List[int] = []

    for instance in instances:
        cells = alterations.get(instance.instance_id)
        if cells is None:
            continue
        for label, design in ((0, instance.map_design), (1, flip_cells(instance.map_design, cells))):
            maps, starts, goals = replace(instance, map_design=design).to_tensors()
            with torch.no_grad():
                cost = encode_cost_map(planner, maps, starts, goals)[0, 0].numpy()
            matrix, mask = feature_builder(instance, design)
            prediction = np.zeros_like(cost, dtype=np.float64)
            prediction[mask] = surrogate.model.predict(matrix)
            residual = np.where(mask, cost - prediction, 0.0)
            statistics = residual_statistics(residual, mask, clean_sd=clean_sd)
            residual_rows.append([statistics[name] for name in RESIDUAL_FEATURES])
            for variant, vector in pooled_activations(
                planner, instance, design, layers=layers
            ).items():
                activation_rows.setdefault(variant, []).append(vector)
            plain = map_only_features(instance, design)
            map_only_rows.append([plain[name] for name in MAP_ONLY_FEATURES])
            labels.append(label)
            ids.append(instance.instance_id)

    return (
        np.asarray(residual_rows, dtype=np.float64),
        {k: np.asarray(v, dtype=np.float64) for k, v in activation_rows.items()},
        np.asarray(map_only_rows, dtype=np.float64),
        np.asarray(labels, dtype=int),
        ids,
    )


def fit_detector(train_features: np.ndarray, train_labels: np.ndarray):
    """Fit the standard detector head: standardise, then logistic regression.

    Standardising matters because the seven residual statistics are on wildly
    different scales — a fraction between 0 and 1 next to a maximum absolute
    residual — and a logistic regression without it would be dominated by
    whichever feature happens to have the largest units.

    Args:
        train_features: Training feature matrix.
        train_labels: 1 for altered, 0 for clean.

    Returns:
        Pipeline: The fitted model, whose ``predict_proba`` gives the score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
    ).fit(train_features, train_labels)


def evaluate_detector(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    model=None,
) -> Dict[str, float]:
    """Fit a logistic regression on the residual features and score it.

    Args:
        train_features: Training feature matrix. Ignored when `model` is given.
        train_labels: 1 for altered, 0 for clean.
        test_features: Evaluation feature matrix.
        test_labels: Evaluation labels.
        model: A head already fitted elsewhere, used instead of fitting a new
            one. This is how the transfer test works: a detector trained on one
            alteration condition is scored on another, which is what `PLAN.md`
            asks for and a much harder test than fitting per condition.

    Returns:
        dict[str, float]: ``auc`` (ranking quality, 0.5 = chance) and
        ``accuracy`` at the default threshold.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    model = model if model is not None else fit_detector(train_features, train_labels)
    scores = model.predict_proba(test_features)[:, 1]
    return {
        "auc": float(roc_auc_score(test_labels, scores)),
        "accuracy": float(accuracy_score(test_labels, model.predict(test_features))),
    }


def evaluate_mahalanobis(
    clean_train: Sequence[np.ndarray], test_vectors: np.ndarray, test_labels: np.ndarray
) -> Dict[str, float]:
    """Score the Mahalanobis baseline, fitted on clean training maps only.

    Args:
        clean_train: Activation vectors of clean training maps.
        test_vectors: Activation vectors to score.
        test_labels: 1 for altered, 0 for clean.

    Returns:
        dict[str, float]: ``auc`` using the distance directly as the score.
    """
    from sklearn.metrics import roc_auc_score

    scorer = MahalanobisScorer.fit(clean_train)
    scores = np.array([scorer.score(vector) for vector in test_vectors])
    return {"auc": float(roc_auc_score(test_labels, scores))}


def bootstrap_auc_gap(
    features_a: np.ndarray,
    features_b: np.ndarray,
    labels: np.ndarray,
    is_train: np.ndarray,
    ids: Sequence[int],
    draws: int = 2000,
    seed: int = 0,
    models: Tuple[object, object] = (None, None),
) -> Dict[str, float]:
    """Confidence interval for the AUC gap between two feature sets.

    With fifty test instances, a gap of a few hundredths of AUC is noise. This
    project has already had one finding overturned by taking a small margin at
    face value, so every claimed gap gets an interval.

    Both models are fitted once on the training half; the bootstrap resamples
    only the *evaluation* instances, which is what the interval is about. Whole
    instances are resampled together, keeping each clean/altered pair intact.

    Args:
        features_a: Feature matrix for the first detector, all examples.
        features_b: Feature matrix for the second detector, same row order.
        labels: 1 for altered, 0 for clean.
        is_train: Which rows belong to the fitting half.
        ids: Instance id per row, used to resample by instance.
        draws: Bootstrap resamples.
        seed: Seed for the resampling.
        models: Heads already fitted elsewhere, one per feature set. Supplied
            by the transfer test, where refitting would defeat the purpose;
            a `None` entry is fitted on the training half as usual.

    Returns:
        dict[str, float]: ``gap`` (A minus B on the full evaluation half),
        ``lo`` and ``hi`` (percentile 95 % interval) and ``p_positive``, the
        share of resamples where A beat B.
    """
    from sklearn.metrics import roc_auc_score

    head_a, head_b = models
    head_a = head_a if head_a is not None else fit_detector(features_a[is_train], labels[is_train])
    head_b = head_b if head_b is not None else fit_detector(features_b[is_train], labels[is_train])
    scores_a = head_a.predict_proba(features_a)[:, 1]
    scores_b = head_b.predict_proba(features_b)[:, 1]

    identifiers = np.asarray(ids)
    evaluation = ~is_train
    unique = np.unique(identifiers[evaluation])
    rng = np.random.default_rng(seed)
    gaps: List[float] = []
    for _ in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([np.flatnonzero(evaluation & (identifiers == i)) for i in picked])
        if len(np.unique(labels[rows])) < 2:
            continue
        gaps.append(
            roc_auc_score(labels[rows], scores_a[rows])
            - roc_auc_score(labels[rows], scores_b[rows])
        )
    array = np.asarray(gaps)
    full = roc_auc_score(labels[evaluation], scores_a[evaluation]) - roc_auc_score(
        labels[evaluation], scores_b[evaluation]
    )
    return {
        "gap": float(full),
        "lo": float(np.percentile(array, 2.5)),
        "hi": float(np.percentile(array, 97.5)),
        "p_positive": float((array > 0).mean()),
    }


def paired_alteration(
    instance: MazeInstance,
    added: RC,
    rng: np.random.Generator,
    polarity: str = "free",
    attempts: int = 200,
) -> Optional[List[RC]]:
    """Add a second flip to an alteration, either freeing or blocking a cell.

    Section 4.2 of the step-12 write-up found that the out-of-distribution
    baseline's strength is, to a correlation of 0.994, obstacle counting — and
    the model-free control's strongest feature is of the same kind. Both should
    be weakened by an attacker who removes as many obstacles as they add.

    Testing that needs two conditions, not one, because an alteration with a
    second flip in it is a bigger alteration regardless of its direction. So
    this builds either variant from the same damaging first flip:

    * ``polarity="free"`` turns an obstacle free, leaving the obstacle count
      unchanged — the balanced attack;
    * ``polarity="block"`` blocks a second free cell, raising the count by two
      — the matched control that holds the number of flips fixed.

    The second cell is drawn from outside the first's 11 × 11 receptive-field
    window, so the two flips do not land in the same patch of the encoder's
    input and the second cannot simply undo the first.

    Args:
        instance: The problem instance, supplying the map, start and goal.
        added: The cell to block, normally a map's worst single flip.
        rng: Source of randomness for choosing the second cell.
        polarity: ``"free"`` or ``"block"``, as above.
        attempts: How many candidates to try before giving up.

    Returns:
        list[tuple[int, int]] | None: The two cells to flip, or None when no
        candidate keeps the goal reachable. This is rare for ``"free"``, since
        freeing a cell cannot disconnect anything, and uncommon but real for
        ``"block"``.

    Raises:
        ValueError: If `polarity` is not one of the two accepted values.
    """
    from .grid import is_reachable

    if polarity not in ("free", "block"):
        raise ValueError(f"polarity must be 'free' or 'block', got {polarity!r}")
    free = instance.map_design > 0
    pool = np.argwhere(free if polarity == "block" else ~free)
    protected = {instance.start_rc, instance.goal_rc, tuple(added)}
    far = [
        (int(r), int(c))
        for r, c in pool
        if max(abs(r - added[0]), abs(c - added[1])) > 5 and (int(r), int(c)) not in protected
    ]
    if not far:
        return None
    for _ in range(attempts):
        second = far[int(rng.integers(len(far)))]
        cells = [tuple(added), second]
        altered = flip_cells(instance.map_design, cells)
        if is_reachable(altered, instance.start_rc, instance.goal_rc):
            return cells
    return None


def unpaired_mask(
    labels: np.ndarray, is_train: np.ndarray, ids: Sequence[int]
) -> np.ndarray:
    """Keep one example per evaluation map, so no map appears clean *and* altered.

    The paired task — show the detector a map's clean and altered version and
    ask it to separate them — is easier than the real one. Because both
    examples come from the same map, every difference between them is the
    alteration, and nothing else. A deployed monitor sees one map arrive and
    must judge it against maps it has never seen.

    This builds that harder task from the same examples: evaluation maps are
    split by position, and each contributes *either* its clean row or its
    altered row, never both. Between-map variation then becomes noise the
    detector has to survive, and the AUC it earns is the realistic one.

    Training rows are all kept — the restriction is about what the detector is
    scored on, not what it learned from.

    Args:
        labels: 1 for altered, 0 for clean.
        is_train: Which rows belong to the fitting half.
        ids: Instance id per row.

    Returns:
        np.ndarray: Boolean mask over rows, True for rows to keep.
    """
    identifiers = np.asarray(ids)
    evaluation = sorted(set(identifiers[~is_train].tolist()))
    # Alternate, so the two groups are interleaved rather than split by index:
    # the maps are ordered and a block split could align with map difficulty.
    altered_only = {value for position, value in enumerate(evaluation) if position % 2}
    keep = np.ones(len(labels), dtype=bool)
    for row, (identifier, label, training) in enumerate(zip(identifiers, labels, is_train)):
        if training:
            continue
        wants_altered = identifier in altered_only
        keep[row] = bool(label) == wants_altered
    return keep


class GridCnn(torch.nn.Module):
    """A small convolutional classifier that sees only the map, never the planner.

    This is the hardest version of the model-free control. The five hand-chosen
    grid statistics of `map_only_features` are a weak baseline by construction:
    somebody had to think of the right five things. A convolutional network
    given the raw grid can discover whatever local pattern an alteration leaves
    — a lone obstacle, a notched wall, a one-cell gap — without being told to
    look for it.

    If this control matches the residual detector, then the planner's internals
    contributed nothing that careful inspection of the map would not have found,
    and the honest conclusion is to say so.

    The architecture deliberately mirrors the thing it is a control for: 3 × 3
    convolutions with the same receptive field growth as the planner's encoder,
    so the two detectors see neighbourhoods of the same size. A global maximum
    over positions makes the output independent of *where* the evidence is,
    which is what lets it generalise from one map to another.

    Attributes:
        body: The convolutional stack.
        head: The linear layer producing a single logit.
    """

    def __init__(self, channels: int = 16) -> None:
        """Build the network.

        Args:
            channels: Width of the convolutional stack.
        """
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(2, channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(2 * channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Score a batch of maps.

        Args:
            inputs: ``(batch, 2, H, W)`` — channel 0 the map, channel 1 the
                start and goal, matching what the planner's encoder receives.

        Returns:
            torch.Tensor: ``(batch,)`` logits, higher meaning "altered".
        """
        features = self.body(inputs)
        pooled = torch.cat(
            [features.amax(dim=(2, 3)), features.mean(dim=(2, 3))], dim=1
        )
        return self.head(pooled)[:, 0]


def grid_cnn_inputs(instances: Sequence[MazeInstance], designs: Sequence[np.ndarray]):
    """Stack maps and endpoint markers into the `GridCnn` input tensor.

    Args:
        instances: One instance per design, supplying the start and goal.
        designs: The (possibly altered) maps.

    Returns:
        torch.Tensor: ``(n, 2, H, W)`` float tensor.
    """
    rows = []
    for instance, design in zip(instances, designs):
        endpoints = np.zeros_like(design, dtype=np.float32)
        endpoints[instance.start_rc] = 1.0
        endpoints[instance.goal_rc] = 1.0
        rows.append(np.stack([design.astype(np.float32), endpoints]))
    return torch.from_numpy(np.stack(rows))


def train_grid_cnn(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    seed: int = 0,
    channels: int = 16,
) -> GridCnn:
    """Fit the convolutional control.

    Args:
        inputs: ``(n, 2, H, W)`` from `grid_cnn_inputs`.
        labels: ``(n,)`` 1 for altered, 0 for clean.
        epochs: Passes over the data.
        batch_size: Examples per step.
        learning_rate: Adam step size.
        seed: Seed for initialisation and shuffling.
        channels: Width passed to `GridCnn`.

    Returns:
        GridCnn: The fitted network, in eval mode.
    """
    torch.manual_seed(seed)
    model = GridCnn(channels=channels)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(inputs), generator=generator)
        for begin in range(0, len(order), batch_size):
            rows = order[begin : begin + batch_size]
            optimiser.zero_grad()
            loss = loss_function(model(inputs[rows]), labels[rows])
            loss.backward()
            optimiser.step()
    model.eval()
    return model


def score_grid_cnn(model: GridCnn, inputs: torch.Tensor) -> np.ndarray:
    """Score maps with the convolutional control.

    Args:
        model: A fitted `GridCnn`.
        inputs: ``(n, 2, H, W)`` from `grid_cnn_inputs`.

    Returns:
        np.ndarray: ``(n,)`` scores, higher meaning "altered".
    """
    with torch.no_grad():
        return model(inputs).numpy()
