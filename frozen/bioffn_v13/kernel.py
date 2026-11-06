"""Full-output residual slots with one fixed, full-input distance calibration.

The 39 coordinates are morphology (16), RNA/response key (16), and condition
(7). AND is the existing V8 RBF without reimplementation. Other views retain
its single amplitude and distance denominator, changing only which distances
are added. Block RBFs are positive semidefinite (PSD); their products are PSD
by the Schur product theorem, and their nonnegative sums remain PSD. Thus AND,
morphology, mechanism, and OR are valid kernels. The separately named softmax
control is normalized positive retrieval, not a PSD-kernel or GP claim.

No data are loaded here. Geometry, residuals and grouping must come from the
authorized fit partition. Excluded-block invariance is conditional on the SAME
frozen full-input geometry; estimating a new denominator is not that control.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from bioffn_v7 import baselines as BL
from bioffn_v8 import memory as V8
from bioffn_v10.memory import _geometry


KINDS = ("and", "morphology", "mechanism", "or", "softmax")
FEATURE_DIM = 39
X_SLICE, RNA_SLICE, CONDITION_SLICE = slice(0, 16), slice(16, 32), slice(32, 39)


def _features(value, name="features"):
    value = BL._matrix(value, name)
    if value.shape[1] != FEATURE_DIM:
        raise ValueError("V13 kernel features must have 39 columns (16 + 16 + 7)")
    return value


def _parameters(kind, bandwidth):
    if kind not in KINDS:
        raise ValueError(f"Unknown residual slot kind: {kind!r}")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be finite and positive")


def _squared_distance(train, query):
    out = np.maximum(np.sum(query ** 2, axis=1)[:, None]
                     + np.sum(train ** 2, axis=1)[None, :] - 2 * query @ train.T, 0.)
    if not np.isfinite(out).all():
        raise ValueError("Feature magnitude overflowed squared distances")
    return out


def kernel_matrix(train, query, kind, geometry, bandwidth=1.):
    """Return a query-by-training Gram matrix, with one shared amplitude.

For softmax this returns its unnormalized full-input positive RBF affinities;
the row normalization and chemical weights are applied by ``attention_weights``.
    """
    train, query = _features(train, "train features"), _features(query, "query features")
    _parameters(kind, bandwidth)
    geo = _geometry(geometry)
    if kind in ("and", "softmax"):
        return V8.kernel_matrix(train, query, "rbf", bandwidth, geo)
    distances = [_squared_distance(train[:, sl], query[:, sl])
                 for sl in (X_SLICE, RNA_SLICE, CONDITION_SLICE)]
    dx, dr, dc = distances
    rate = bandwidth / geo["distance_reference"]
    if kind == "morphology":
        radial = np.exp(-rate * (dx + dc))
    elif kind == "mechanism":
        radial = np.exp(-rate * (dr + dc))
    else:
        radial = .5 * (np.exp(-rate * dx) + np.exp(-rate * dr)) * np.exp(-rate * dc)
    return geo["linear_scale"] * radial


@dataclass
class SlotModel:
    train_features: np.ndarray
    dual_coefficients: np.ndarray | None
    intercept: np.ndarray
    centered_residual: np.ndarray
    chemical_row_weights: np.ndarray
    geometry: dict[str, Any]
    kind: str
    bandwidth: float
    audit: dict[str, Any]

    def attention_weights(self, query):
        """Positive chemical-weighted retrieval probabilities (softmax only)."""
        if self.kind != "softmax":
            raise ValueError("Attention weights are only defined for the softmax control")
        query = _features(query, "query features")
        distances = _squared_distance(self.train_features, query)
        distances -= np.min(distances, axis=1, keepdims=True)
        logits = (-self.bandwidth * distances / self.geometry["distance_reference"]
                  + np.log(self.chemical_row_weights)[None, :])
        # Subtract the nearest-neighbour distance before logsumexp to avoid
        # losing the small log(group weight) term when all distances are huge.
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return weights

    def predict(self, query):
        query = _features(query, "query features")
        if self.kind == "softmax":
            return self.intercept + self.attention_weights(query) @ self.centered_residual
        return self.intercept + kernel_matrix(self.train_features, query, self.kind,
                                              self.geometry, self.bandwidth) @ self.dual_coefficients

    def to_dict(self):
        return {"format": "v13_residual_slot", "train_features": self.train_features.tolist(),
                "dual_coefficients": None if self.dual_coefficients is None else self.dual_coefficients.tolist(),
                "intercept": self.intercept.tolist(), "centered_residual": self.centered_residual.tolist(),
                "chemical_row_weights": self.chemical_row_weights.tolist(),
                "geometry": copy.deepcopy(self.geometry), "kind": self.kind,
                "bandwidth": self.bandwidth, "audit": copy.deepcopy(self.audit)}

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v13_residual_slot":
            raise ValueError("Unknown residual-slot format")
        features = _features(saved["train_features"], "saved features").copy()
        target = BL._matrix(saved["centered_residual"], "saved centered residual", len(features)).copy()
        mean = np.asarray(saved["intercept"], float)
        weights = np.asarray(saved["chemical_row_weights"], float)
        kind, bandwidth = saved["kind"], float(saved["bandwidth"])
        _parameters(kind, bandwidth)
        if mean.shape != (target.shape[1],) or not np.isfinite(mean).all():
            raise ValueError("Saved intercept must match the full output dimension")
        if (weights.shape != (len(features),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.)):
            raise ValueError("Saved chemical weights must be positive and sum to one")
        dual = saved["dual_coefficients"]
        if kind == "softmax":
            if dual is not None:
                raise ValueError("Softmax control cannot have ridge coefficients")
        else:
            dual = BL._matrix(dual, "saved dual coefficients", len(features)).copy()
            if dual.shape != target.shape:
                raise ValueError("Saved dual coefficients differ from the output dimension")
        return cls(features, dual, mean.copy(), target, weights.copy(),
                   _geometry(saved["geometry"]), kind, bandwidth, copy.deepcopy(saved["audit"]))


def fit_slot(features, residual, groups, kind="and", geometry=None, lambda_value=100.,
             bandwidth=1.):
    """Fit a full-output residual slot; its audit is available as ``model.audit``.

KRR uses W=n*group_weights, fixed chemical-equal intercept m and solves
``(sqrt(W) K sqrt(W) + lambda I)b=sqrt(W)(R-m)``, alpha=sqrt(W)b.
Softmax has no fitted regression coefficients and ignores ridge lambda.
    """
    features = _features(features)
    residual = BL._matrix(residual, "residual", len(features))
    groups = BL._labels(groups, "groups", len(features))
    _parameters(kind, bandwidth)
    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("Ridge lambda must be finite and positive")
    geo = _geometry(V8.kernel_geometry(features) if geometry is None else geometry)
    weights = BL.group_weights(groups)
    intercept = weights @ residual
    centered = residual - intercept
    alpha = None
    audit = {"kind": kind, "rows": len(features), "chemical_groups": len(np.unique(groups)),
             "feature_dim": FEATURE_DIM, "full_output_dim": residual.shape[1],
             "feature_slices": {"morphology": [0, 16], "mechanism": [16, 32], "condition": [32, 39]},
             "geometry": copy.deepcopy(geo), "geometry_explicit": geometry is not None,
             "geometry_scope": "same full-input FIT distance reference and amplitude for all views",
             "bandwidth": float(bandwidth), "lambda": float(lambda_value),
             "lambda_used": kind != "softmax", "intercept_scope": "chemical-equal FIT residual mean",
             "chemical_row_weights": weights.tolist(), "data_row_weights": (len(features) * weights).tolist(),
             "data_weight_sum": float(len(features) * weights.sum()),
             "psd_kernel_claim": kind != "softmax"}
    if kind == "softmax":
        audit.update({"solver": "positive row-normalized RBF retrieval with chemical weights",
                      "objective": "none; fixed-bandwidth normalized retrieval, not KRR",
                      "relative_normal_equation_error": None})
    else:
        sw = np.sqrt(len(features) * weights)
        gram = kernel_matrix(features, features, kind, geo, bandwidth)
        system = sw[:, None] * gram * sw[None, :]
        system = (system + system.T) * .5
        system.flat[::len(system) + 1] += lambda_value
        factor = cho_factor(system, lower=True, check_finite=True)
        scaled = cho_solve(factor, sw[:, None] * centered, check_finite=True)
        alpha = np.ascontiguousarray(sw[:, None] * scaled)
        error = system @ scaled - sw[:, None] * centered
        audit.update({"solver": "symmetric weighted Cholesky; no added jitter",
                      "objective": "n sum group_weights ||R-m-f||² + lambda ||f||_H²",
                      "relative_normal_equation_error": float(np.linalg.norm(error) /
                          max(1., float(np.linalg.norm(sw[:, None] * centered))))})
    model = SlotModel(features.copy(), alpha, intercept.copy(), centered.copy(), weights.copy(),
                      geo, kind, float(bandwidth), audit)
    return model


def wholecompound_permutation(features, groups, codes, seed, rna_slice=RNA_SLICE):
    """Permute complete molecular trajectories, retaining morphology/conditions.

Donors are restricted to identical condition multisets. Each eligible stratum
uses a random ordering followed by a nonzero cyclic shift, a derangement but
not a uniform draw over all possible derangements. All rows of a recipient
share the same donor. Repeated occurrences of one condition match in input
row order; unique-condition trajectories are also invariant to row ordering.
Singleton strata remain unchanged and are explicitly reported. No residual or
target is accepted. These unconditional within-inventory permutations do not
assert conditional exchangeability after controlling morphology.
    """
    features = _features(features)
    groups = BL._labels(groups, "groups", len(features))
    codes = BL._labels(codes, "codes", len(features))
    if not isinstance(rna_slice, slice):
        raise ValueError("RNA columns must be a contiguous slice")
    start, stop, step = rna_slice.indices(FEATURE_DIM)
    if step != 1 or stop <= start:
        raise ValueError("RNA columns must be a nonempty contiguous slice")
    selected = slice(start, stop)
    strata = {}
    for group in np.unique(groups):
        vals, counts = np.unique(codes[groups == group], return_counts=True)
        signature = tuple(zip(vals.tolist(), counts.tolist()))
        strata.setdefault(signature, []).append(group)
    if not any(len(members) > 1 for members in strata.values()):
        raise ValueError("No compatible compound trajectory stratum permits a nontrivial permutation")
    rng = np.random.default_rng(seed)
    donor_map, stratum_audits = {}, []
    for signature, members in sorted(strata.items()):
        members = np.asarray(sorted(members))
        if len(members) == 1:
            donor_map[str(members[0])] = str(members[0])
        else:
            ordered = rng.permutation(members)
            donors = np.roll(ordered, int(rng.integers(1, len(ordered))))
            donor_map.update({str(a): str(b) for a, b in zip(ordered, donors)})
        stratum_audits.append({"condition_multiset": [[k, v] for k, v in signature],
                               "groups": members.tolist(), "singleton": len(members) == 1})
    donor_rows = np.empty(len(features), dtype=int)
    for group, donor in donor_map.items():
        for code in np.unique(codes[groups == group]):
            recipient_indices = np.flatnonzero((groups == group) & (codes == code))
            source_indices = np.flatnonzero((groups == donor) & (codes == code))
            donor_rows[recipient_indices] = source_indices
    result = features.copy()
    result[:, selected] = features[donor_rows, selected]
    audit = {"seed": int(seed), "donor_by_group": donor_map, "donor_row_indices": donor_rows.tolist(),
             "rna_columns": [start, stop], "condition_matching": "exact condition multiset",
             "duplicate_condition_matching": "occurrence order within input",
             "permutation_algorithm": "random ordering plus nonzero cyclic shift within inventory stratum",
             "eligible_groups_changed": sum(a != b for a, b in donor_map.items()),
             "unchanged_singleton_groups": [a for a, b in donor_map.items() if a == b],
             "strata": stratum_audits, "target_accessed": False,
             "conditional_exchangeability_claim": False}
    return result, audit
