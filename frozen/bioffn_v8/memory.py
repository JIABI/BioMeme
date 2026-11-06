"""Fold-local, full-output residual memory with a fixed copy baseline.

For a fixed external predictor E, every candidate is
    G(x,c) + s E(Z_r) + m + k(z,Z) alpha.
G is always fitted to the original y, never to y-sE. The memory targets are
honest training-OOF residuals y-G_OOF-sE. Z_r uses the destination training
fold's RNA moments: this is explicitly domain-standardized transfer.

No project data are loaded here. A prior exposes predict_standardized(Z_r).
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from bioffn_v7 import baselines as BL
from bioffn_v7.features import _pca


DEFAULT_WEIGHTS = (0., .1, .3, 1., 3., "rbf")
DEFAULT_BANDWIDTHS = (.25, 1., 4.)
_EPS = 1e-12


def _inputs(x, rna, conditions, codes, groups=None):
    x = BL._matrix(x, "x")
    rna = BL._matrix(rna, "rna", len(x))
    conditions = BL._matrix(conditions, "conditions", len(x))
    codes = BL._labels(codes, "codes", len(x))
    if groups is None:
        return x, rna, conditions, codes
    return x, rna, conditions, codes, BL._labels(groups, "groups", len(x))


@dataclass
class MemoryCoordinates:
    """PCA blocks preserve relative PC variance; conditions use column scales."""
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_components: np.ndarray
    x_block_scale: float
    rna_mean: np.ndarray
    rna_scale: np.ndarray
    rna_components: np.ndarray
    rna_block_scale: float
    condition_mean: np.ndarray
    condition_scale: np.ndarray
    include_rna: bool
    audit: dict[str, Any]

    @classmethod
    def fit(cls, x, rna, conditions, groups, *, x_rank=16, rna_rank=16,
            include_rna=True):
        x = BL._matrix(x, "x")
        rna = BL._matrix(rna, "rna", len(x))
        conditions = BL._matrix(conditions, "conditions", len(x))
        groups = BL._labels(groups, "groups", len(x))
        if len(x) < 2:
            raise ValueError("PCA needs at least two training rows")
        for rank in (x_rank, rna_rank):
            if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
                raise ValueError("PCA ranks must be positive integers")
        w = BL.group_weights(groups)
        xm, xs, xp, xb, xa = _pca(x, w, x_rank)
        rm, rs, rp, rb, ra = _pca(rna, w, rna_rank)
        cm = w @ conditions
        cs = np.sqrt(np.maximum(w @ np.square(conditions - cm), 0.))
        constant = cs <= _EPS
        cs[constant] = 1.
        audit = {"fit_rows": len(x), "fit_groups": np.unique(groups).tolist(),
                 "x_pca": xa, "rna_pca": ra,
                 "condition_constant_columns": np.flatnonzero(constant).tolist(),
                 "include_rna_in_memory": bool(include_rna),
                 "summary": "x PCA / one block RMS; RNA PCA / one block RMS; column-standardized conditions",
                 "prior_coordinates": "destination TRAIN gene mean/std, before RNA PCA",
                 "targets_used": False}
        return cls(xm, xs, xp, xb, rm, rs, rp, rb, cm, cs, bool(include_rna), audit)

    def standardize_rna(self, rna):
        rna = BL._matrix(rna, "rna")
        if rna.shape[1] != len(self.rna_mean):
            raise ValueError("RNA input dimension differs from fitted coordinates")
        return (rna - self.rna_mean) / self.rna_scale

    def transform(self, x, rna, conditions):
        x = BL._matrix(x, "x")
        rna = BL._matrix(rna, "rna", len(x))
        conditions = BL._matrix(conditions, "conditions", len(x))
        if x.shape[1] != len(self.x_mean) or conditions.shape[1] != len(self.condition_mean):
            raise ValueError("Input dimension differs from fitted coordinates")
        zr = self.standardize_rna(rna)
        blocks = [((x - self.x_mean) / self.x_scale) @ self.x_components / self.x_block_scale]
        if self.include_rna:
            blocks.append(zr @ self.rna_components / self.rna_block_scale)
        blocks.append((conditions - self.condition_mean) / self.condition_scale)
        return np.concatenate(blocks, axis=1)

    def to_dict(self):
        result = {name: value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
                  for name, value in vars(self).items()}
        result["format"] = "v8_memory_coordinates"
        return result

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        if value.pop("format", None) != "v8_memory_coordinates":
            raise ValueError("Unknown coordinate format")
        for name in ("x_mean", "x_scale", "x_components", "rna_mean", "rna_scale",
                     "rna_components", "condition_mean", "condition_scale"):
            value[name] = np.asarray(value[name], dtype=float)
            if not np.isfinite(value[name]).all():
                raise ValueError("Non-finite coordinate parameters")
        if any(np.any(value[k] <= 0) for k in ("x_scale", "rna_scale", "condition_scale")):
            raise ValueError("Coordinate scales must be positive")
        if value["x_block_scale"] <= 0 or value["rna_block_scale"] <= 0:
            raise ValueError("PCA block scales must be positive")
        return cls(**value)


def _prior(prior_predictor, coordinates, rna, outputs):
    if prior_predictor is None:
        return np.zeros((len(rna), outputs))
    pred = BL._matrix(prior_predictor.predict_standardized(coordinates.standardize_rna(rna)),
                      "prior prediction", len(rna))
    if pred.shape[1] != outputs:
        raise ValueError("Prior must return the full output dimension")
    return pred


def prepare_memory_split(x, y, rna, conditions, codes, groups, train_indices,
                         valid_indices, prior_predictor=None, *, folds=8,
                         seed=20261008, x_rank=16, rna_rank=16, include_rna=True):
    """Construct one selection fold without reading its validation targets.

    Perturbing outer validation targets cannot alter any returned upstream
    quantity. Validation inputs are transformed, never used to estimate moments.
    """
    x, rna, conditions, codes, groups = _inputs(x, rna, conditions, codes, groups)
    y = np.asarray(y, dtype=float)
    if y.shape != x.shape:
        raise ValueError("Copy calibration requires matching source and output features")
    train = BL._indices(train_indices, len(x), "train_indices")
    valid = BL._indices(valid_indices, len(x), "valid_indices")
    if set(groups[train]) & set(groups[valid]):
        raise ValueError("Training and validation chemicals overlap")
    BL._matrix(y[train], "training targets")
    base, ba = BL.fit_calibration(x[train], y[train], codes[train], groups[train],
                                  mode="scalar", n_folds=folds, seed=seed)
    oof, oa = BL.crossfit_calibration(x[train], y[train], codes[train], groups[train],
                                     mode="scalar", n_folds=folds, seed=seed + 7919)
    coords = MemoryCoordinates.fit(x[train], rna[train], conditions[train], groups[train],
        x_rank=x_rank, rna_rank=rna_rank, include_rna=include_rna)
    return {"base": base, "coordinates": coords,
            "residual_train": y[train] - oof,
            "X_train": coords.transform(x[train], rna[train], conditions[train]),
            "X_valid": coords.transform(x[valid], rna[valid], conditions[valid]),
            "base_valid": base.predict(x[valid], codes[valid]),
            "prior_train": _prior(prior_predictor, coords, rna[train], y.shape[1]),
            "prior_valid": _prior(prior_predictor, coords, rna[valid], y.shape[1]),
            "audit": {"train_indices": train.tolist(), "validation_indices": valid.tolist(),
                      "train_groups": np.unique(groups[train]).tolist(),
                      "validation_groups": np.unique(groups[valid]).tolist(),
                      "baseline": ba, "residual_crossfit": oa,
                      "coordinates": coords.audit, "baseline_target": "original y",
                      "validation_targets_used_upstream": False}}


def kernel_geometry(features):
    features = BL._matrix(features, "features")
    squared = np.sum(features * features, axis=1)
    distances = np.maximum(squared[:, None] + squared[None, :] - 2 * features @ features.T, 0.)
    off = distances[~np.eye(len(features), dtype=bool)]
    median = float(np.median(off)) if len(off) else 0.
    scale = float(np.mean(squared))
    return {"distance_reference": median if median > _EPS else 1.,
            "linear_scale": scale if scale > _EPS else 1.,
            "distance_degenerate": median <= _EPS, "linear_degenerate": scale <= _EPS}


def kernel_matrix(train, query, weight, bandwidth, geometry):
    """Return query x train; nonnegative mixtures and pure RBF are PSD."""
    train, query = np.asarray(train, float), np.asarray(query, float)
    if train.ndim != 2 or query.ndim != 2 or train.shape[1] != query.shape[1]:
        raise ValueError("Kernel features are not aligned matrices")
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("Kernel inputs must be finite")
    if weight != "rbf" and (not np.isfinite(weight) or weight < 0):
        raise ValueError("Kernel weight must be nonnegative or 'rbf'")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth multiplier must be positive")
    linear = query @ train.T
    if weight == 0:
        return linear
    d2 = np.maximum(np.sum(query ** 2, 1)[:, None] + np.sum(train ** 2, 1)[None, :] - 2 * linear, 0.)
    radial = geometry["linear_scale"] * np.exp(-bandwidth * d2 / geometry["distance_reference"])
    return radial if weight == "rbf" else linear + weight * radial


class _KernelPath:
    """One eigendecomposition; no full prediction grid is retained."""
    def __init__(self, features, groups, weight, bandwidth, geometry=None):
        self.features = np.asarray(features, float)
        self.geometry = kernel_geometry(features) if geometry is None else dict(geometry)
        self.weight, self.bandwidth = weight, bandwidth
        self.weights = BL.group_weights(groups)
        self.sqrt_weight = np.sqrt(len(features) * self.weights)
        K = kernel_matrix(features, features, weight, bandwidth, self.geometry)
        Kw = self.sqrt_weight[:, None] * K * self.sqrt_weight[None, :]
        values, vectors = np.linalg.eigh((Kw + Kw.T) * .5)
        tolerance = 1e-8 * max(1., float(np.max(np.abs(values))))
        if values.min() < -tolerance:
            raise ArithmeticError("Kernel has materially negative eigenvalues")
        self.values, self.vectors = np.maximum(values, 0.), vectors

    def transfer(self, query, lam):
        Kn = kernel_matrix(self.features, query, self.weight, self.bandwidth, self.geometry)
        left = (Kn * self.sqrt_weight[None, :]) @ self.vectors
        return (left / (self.values + lam)[None, :]) @ self.vectors.T

    def solve(self, residual, lam):
        mean = self.weights @ residual
        proj = self.vectors.T @ (self.sqrt_weight[:, None] * (residual - mean))
        alpha = self.sqrt_weight[:, None] * (self.vectors @ (proj / (self.values + lam)[:, None]))
        return alpha, mean


def _risk_stats(residual, error_valid_without_mean, train_weights, valid_weights):
    """Exact full-output CV loss statistics, not an output-space compression."""
    mean = train_weights @ residual
    A = np.sqrt(len(residual) * train_weights)[:, None] * (residual - mean)
    E = error_valid_without_mean - mean
    p = residual.shape[1]
    return {"gram": A @ A.T / p, "cross": A @ E.T / p,
            "error": float(valid_weights @ np.mean(E * E, axis=1)), "weights": valid_weights}


def _risk_from_transfer(transfer, stats):
    quadratic = np.sum((transfer @ stats["gram"]) * transfer, axis=1)
    cross = np.sum(transfer * stats["cross"].T, axis=1)
    value = stats["error"] + float(stats["weights"] @ (quadratic - 2 * cross))
    if value < -1e-8 * max(1., stats["error"]):
        raise ArithmeticError("Negative MSE beyond roundoff")
    return max(0., value)


def _candidate(scale, memory=False, weight=None, bandwidth=None, lam=None):
    return {"scale": float(scale), "memory": bool(memory), "weight": weight,
            "bandwidth": bandwidth, "lambda": lam}


def _choose_index(curve, candidates):
    # Exact ties prefer no memory, no atlas, then stronger ridge regularization.
    return min(range(len(curve)), key=lambda i: (curve[i], candidates[i]["memory"],
        abs(candidates[i]["scale"]), -(candidates[i]["lambda"] or 0.)))


@dataclass
class MemoryModel:
    base: BL.Calibration
    coordinates: MemoryCoordinates
    train_features: np.ndarray
    dual_coefficients: np.ndarray
    residual_mean: np.ndarray
    selected: dict[str, Any]
    geometry: dict[str, Any]
    prior_predictor: Any
    audit: dict[str, Any]

    def predict(self, x, rna, conditions, codes):
        x, rna, conditions, codes = _inputs(x, rna, conditions, codes)
        prediction = self.base.predict(x, codes)
        if self.selected["scale"]:
            prediction = prediction + self.selected["scale"] * _prior(
                self.prior_predictor, self.coordinates, rna, self.base.n_features)
        if self.selected["memory"]:
            features = self.coordinates.transform(x, rna, conditions)
            K = kernel_matrix(self.train_features, features, self.selected["weight"],
                              self.selected["bandwidth"], self.geometry)
            prediction = prediction + K @ self.dual_coefficients + self.residual_mean
        return prediction

    def to_dict(self):
        return {"format": "v8_memory_model", "base": self.base.to_dict(),
                "coordinates": self.coordinates.to_dict(),
                "train_features": self.train_features.tolist(),
                "dual_coefficients": self.dual_coefficients.tolist(),
                "residual_mean": self.residual_mean.tolist(),
                "selected": copy.deepcopy(self.selected), "geometry": dict(self.geometry),
                "prior_required": bool(self.selected["scale"]),
                "audit": copy.deepcopy(self.audit)}

    @classmethod
    def from_dict(cls, value, prior_predictor=None):
        if value.get("format") != "v8_memory_model":
            raise ValueError("Unknown memory model format")
        if value["prior_required"] and prior_predictor is None:
            raise ValueError("Saved model requires its external prior predictor")
        base = BL.Calibration.from_dict(value["base"])
        coords = MemoryCoordinates.from_dict(value["coordinates"])
        features = BL._matrix(value["train_features"], "train_features")
        alpha = BL._matrix(value["dual_coefficients"], "dual_coefficients", len(features))
        mean = np.asarray(value["residual_mean"], float)
        if alpha.shape[1] != base.n_features or mean.shape != (base.n_features,) or not np.isfinite(mean).all():
            raise ValueError("Saved full-output dimensions disagree")
        return cls(base, coords, features, alpha, mean, dict(value["selected"]),
                   dict(value["geometry"]), prior_predictor, copy.deepcopy(value["audit"]))

    def save(self, path):
        with Path(path).open("w") as handle:
            json.dump(self.to_dict(), handle)


def fit_memory(x, y, rna, conditions, codes, groups, prior_predictor=None, *,
               kind="kernel", seed=20261008, folds=8, scale_grid=(0., .5, 1.),
               include_rna=True, x_rank=16, rna_rank=16,
               lambdas=BL.DEFAULT_LAMBDAS, kernel_weights=DEFAULT_WEIGHTS,
               bandwidths=DEFAULT_BANDWIDTHS):
    """Select on grouped FIT CV only; refit once using original-y G.

    The external prior is assumed already fitted on a disjoint authorized atlas.
    It is never fitted or mutated here. No TUNE/evaluation argument exists.
    """
    x, rna, conditions, codes, groups = _inputs(x, rna, conditions, codes, groups)
    y = BL._matrix(y, "y", len(x))
    if y.shape != x.shape:
        raise ValueError("Copy calibration requires matching source and output features")
    if len(np.unique(groups)) < 3:
        raise ValueError("Nested residual CV needs at least three chemical groups")
    if kind not in ("kernel", "linear"):
        raise ValueError("kind must be kernel or linear")
    scales = sorted({0., *map(float, scale_grid)}) if prior_predictor is not None else [0.]
    if any(not np.isfinite(s) or s < 0 for s in scales):
        raise ValueError("Prior scales must be finite and nonnegative")
    lams = list(dict.fromkeys(map(float, lambdas)))
    if not lams or any(not np.isfinite(v) or v <= 0 for v in lams):
        raise ValueError("Ridge lambdas must be positive and finite")
    weights = (0.,) if kind == "linear" else tuple(dict.fromkeys(kernel_weights))
    if not weights or any(v != "rbf" and (not np.isfinite(v) or v < 0) for v in weights):
        raise ValueError("Kernel weights must be nonnegative or 'rbf'")
    bands = tuple(dict.fromkeys(map(float, bandwidths)))
    if not bands or any(not np.isfinite(v) or v <= 0 for v in bands):
        raise ValueError("Bandwidth multipliers must be positive and finite")
    kernels = [(w, b) for w in weights for b in ((1.,) if w == 0 else bands)]
    candidates = [_candidate(s) for s in scales]
    candidates += [_candidate(s, True, w, b, lam) for w, b in kernels for lam in lams for s in scales]
    curve = np.zeros(len(candidates))
    fid = BL.make_folds(groups, folds, seed)
    audits = []
    total_groups = len(np.unique(groups))
    for fold in np.unique(fid):
        train, valid = np.flatnonzero(fid != fold), np.flatnonzero(fid == fold)
        prepared = prepare_memory_split(x, y, rna, conditions, codes, groups, train, valid,
            prior_predictor, folds=folds, seed=seed + 104729 * (int(fold) + 1),
            x_rank=x_rank, rna_rank=rna_rank, include_rna=include_rna)
        fw = len(np.unique(groups[valid])) / total_groups
        tw, vw = BL.group_weights(groups[train]), BL.group_weights(groups[valid])
        stats = {}
        risks = np.empty(len(candidates))
        for i, scale in enumerate(scales):
            pred = prepared["base_valid"] + scale * prepared["prior_valid"]
            risks[i] = BL.group_mse(pred, y[valid], groups[valid])
            stats[scale] = _risk_stats(prepared["residual_train"] - scale * prepared["prior_train"],
                                      y[valid] - pred, tw, vw)
        cursor = len(scales)
        geometry = kernel_geometry(prepared["X_train"])
        for weight, band in kernels:
            path = _KernelPath(prepared["X_train"], groups[train], weight, band, geometry)
            for lam in lams:
                transfer = path.transfer(prepared["X_valid"], lam)
                for scale in scales:
                    risks[cursor] = _risk_from_transfer(transfer, stats[scale])
                    cursor += 1
        curve += fw * risks
        audits.append({"fold": int(fold), **prepared["audit"],
                       "kernel_geometry": geometry, "candidate_mse": risks.tolist()})
    best = _choose_index(curve, candidates)
    selected = copy.deepcopy(candidates[best])
    base, ba = BL.fit_calibration(x, y, codes, groups, mode="scalar", n_folds=folds,
                                  seed=seed + 200003)
    oof, oa = BL.crossfit_calibration(x, y, codes, groups, mode="scalar", n_folds=folds,
                                     seed=seed + 200009)
    coords = MemoryCoordinates.fit(x, rna, conditions, groups, x_rank=x_rank,
                                   rna_rank=rna_rank, include_rna=include_rna)
    features = coords.transform(x, rna, conditions)
    geometry = kernel_geometry(features)
    if selected["memory"]:
        residual = y - oof - selected["scale"] * _prior(prior_predictor, coords, rna, y.shape[1])
        path = _KernelPath(features, groups, selected["weight"], selected["bandwidth"], geometry)
        alpha, mean = path.solve(residual, selected["lambda"])
    else:
        alpha, mean = np.zeros((len(x), y.shape[1])), np.zeros(y.shape[1])
    audit = {"method": "v8 fixed-baseline nested residual memory", "kind": kind,
             "fit_rows": len(x), "fit_groups": np.unique(groups).tolist(), "seed": int(seed),
             "actual_outer_folds": len(audits), "outer_folds": audits,
             "candidates": [{**c, "mse": float(v)} for c, v in zip(candidates, curve)],
             "selected": selected, "selected_cv_mse": float(curve[best]),
             "baseline": ba, "final_residual_crossfit": oa, "coordinates": coords.audit,
             "baseline_target": "original y; identical across prior/memory choices",
             "residual_target": "y - group-OOF G(original y) - s * external_prior(destination-FIT-standardized RNA)",
             "loss": "chemical-equal original-coordinate MSE, all output features",
             "ridge_penalty": "sum_i n*w_i ||e_i||^2 + lambda ||f||_H^2, sum(w)=1",
             "intercept": "training residual mean after subtracting the atlas prior; fixed before KRR",
             "selection_scope": "FIT nested grouped CV only; no TUNE or evaluation targets",
             "zero_baseline_candidate_included": True, "full_output": True,
             "include_rna_in_memory": bool(include_rna), "prior_present": prior_predictor is not None,
             "prior_fit_scope": "external predictor supplied by caller; never fitted here"}
    return MemoryModel(base, coords, features, alpha, mean, selected, geometry, prior_predictor, audit)
