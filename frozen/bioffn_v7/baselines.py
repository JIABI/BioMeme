"""Raw-coordinate calibration and nested, chemical-grouped residual ridge.

No dataset is loaded by this module. Every inner selection split rebuilds its
baseline and its training residuals without the outer validation targets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


DEFAULT_SEED = 20261008
DEFAULT_SHRINK_GRID = (0., 1., 3., 10., 30., 100., 300., 1000., 10000., 1e9)
DEFAULT_LAMBDAS = (10., 30., 100., 300., 1000., 3000., 1e4, 3e4, 1e5, 3e5, 1e6)


def _matrix(value, name, rows=None):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not len(value) or value.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if rows is not None and len(value) != rows:
        raise ValueError(f"{name} has the wrong number of rows")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _labels(value, name, rows=None):
    value = np.asarray(value)
    if value.ndim != 1 or not len(value) or (rows is not None and len(value) != rows):
        raise ValueError(f"{name} must be a nonempty aligned vector")
    return value.astype(str)


def _calibration_arrays(x, y, codes, groups):
    x = _matrix(x, "x")
    y = _matrix(y, "y", len(x))
    if x.shape != y.shape:
        raise ValueError("Copy calibration requires matching source and output features")
    return x, y, _labels(codes, "codes", len(x)), _labels(groups, "groups", len(x))


def group_weights(groups):
    """Normalized row weights; every chemical receives equal total mass."""
    groups = _labels(groups, "groups")
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def group_mse(prediction, target, groups):
    target = _matrix(target, "target")
    prediction = _matrix(prediction, "prediction", len(target))
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    groups = _labels(groups, "groups", len(target))
    return float(group_weights(groups) @ np.mean((prediction - target) ** 2, axis=1))


def make_folds(groups, n_folds=8, seed=DEFAULT_SEED):
    """Assign whole chemicals to deterministic, nearly equal-sized folds."""
    groups = _labels(groups, "groups")
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("Cross-fitting needs at least two chemical groups")
    effective = min(int(n_folds), len(unique))
    permutation = np.random.default_rng(seed).permutation(unique)
    assignment = {group: i % effective for i, group in enumerate(permutation)}
    return np.asarray([assignment[group] for group in groups], dtype=np.int64)


class Calibration:
    """Condition-specific affine copy, with scalar or shrunk diagonal slopes.

    Means are chemical-equal within each condition. The diagonal shrinkage
    acts on weighted *sums* (weights sum to the condition's row count), matching
    the original sum-of-squares convention for balanced chemical trajectories.
    """

    def fit(self, x, y, codes, groups, mode="scalar", shrink=0.):
        x, y, codes, groups = _calibration_arrays(x, y, codes, groups)
        if mode not in {"scalar", "diagonal", "diag"}:
            raise ValueError("mode must be scalar or diagonal")
        if not np.isfinite(shrink) or shrink < 0:
            raise ValueError("shrink must be finite and nonnegative")
        self.mode = "diagonal" if mode == "diag" else mode
        self.shrink = float(shrink)
        self.n_features = x.shape[1]
        self.source_mean, self.target_mean, self.slopes, self.scalar_slopes = {}, {}, {}, {}
        for code in np.unique(codes):
            mask = codes == code
            w = group_weights(groups[mask])
            mx, my = w @ x[mask], w @ y[mask]
            xc, yc = x[mask] - mx, y[mask] - my
            sw = len(w) * w
            xx = sw @ (xc * xc)
            xy = sw @ (xc * yc)
            denominator = float(xx.sum())
            scalar = float(xy.sum() / denominator) if denominator > 1e-12 else 0.
            if self.mode == "scalar":
                slope = np.full(self.n_features, scalar)
            else:
                den = xx + self.shrink
                slope = np.full(self.n_features, scalar)
                np.divide(xy + self.shrink * scalar, den, out=slope, where=den > 1e-12)
            self.source_mean[code], self.target_mean[code] = mx, my
            self.slopes[code], self.scalar_slopes[code] = slope, scalar
        return self

    def predict(self, x, codes):
        if not hasattr(self, "n_features"):
            raise ValueError("Calibration has not been fitted")
        x = _matrix(x, "x")
        codes = _labels(codes, "codes", len(x))
        if x.shape[1] != self.n_features:
            raise ValueError("Source feature dimension does not match calibration")
        unseen = set(codes) - set(self.source_mean)
        if unseen:
            raise ValueError(f"Unseen conditions: {sorted(unseen)}")
        result = np.empty_like(x)
        for code in np.unique(codes):
            mask = codes == code
            result[mask] = self.target_mean[code] + self.slopes[code] * (x[mask] - self.source_mean[code])
        return result

    def to_dict(self):
        if not hasattr(self, "n_features"):
            raise ValueError("Calibration has not been fitted")
        return {"format": "v7_calibration", "mode": self.mode, "shrink": self.shrink,
                "n_features": self.n_features,
                "source_mean": {k: v.tolist() for k, v in self.source_mean.items()},
                "target_mean": {k: v.tolist() for k, v in self.target_mean.items()},
                "slopes": {k: v.tolist() for k, v in self.slopes.items()},
                "scalar_slopes": dict(self.scalar_slopes)}

    @classmethod
    def from_dict(cls, value):
        if value.get("format") != "v7_calibration":
            raise ValueError("Unknown calibration format")
        model = cls()
        model.mode, model.shrink = value["mode"], float(value["shrink"])
        model.n_features = int(value["n_features"])
        for field in ("source_mean", "target_mean", "slopes"):
            setattr(model, field, {str(k): np.asarray(v, dtype=np.float64)
                                   for k, v in value[field].items()})
        model.scalar_slopes = {str(k): float(v) for k, v in value["scalar_slopes"].items()}
        keys = set(model.source_mean)
        if not keys or any(set(getattr(model, field)) != keys for field in ("target_mean", "slopes", "scalar_slopes")):
            raise ValueError("Calibration condition keys disagree")
        if model.mode not in {"scalar", "diagonal"} or model.n_features < 1 or not np.isfinite(model.shrink) or model.shrink < 0:
            raise ValueError("Invalid saved calibration configuration")
        if any(v.shape != (model.n_features,) or not np.isfinite(v).all()
               for field in ("source_mean", "target_mean", "slopes") for v in getattr(model, field).values()):
            raise ValueError("Invalid saved calibration vectors")
        if not all(math.isfinite(v) for v in model.scalar_slopes.values()):
            raise ValueError("Invalid saved calibration scalar slopes")
        return model


def fit_calibration(x, y, codes, groups, mode="scalar", shrink=None, n_folds=8,
                    seed=DEFAULT_SEED, shrink_grid=DEFAULT_SHRINK_GRID):
    """Fit a baseline; any diagonal shrinkage selection uses these rows only."""
    x, y, codes, groups = _calibration_arrays(x, y, codes, groups)
    if mode == "diag":
        mode = "diagonal"
    if mode not in {"scalar", "diagonal"}:
        raise ValueError("mode must be scalar or diagonal")
    audit = {"mode": mode, "fit_groups": np.unique(groups).tolist(), "fit_rows": len(x),
             "chemical_equal": True, "selection": "none", "cv": []}
    if mode == "scalar":
        chosen = 0.
    elif shrink is not None:
        chosen = float(shrink)
    else:
        grid = [float(v) for v in shrink_grid]
        if not grid or any(not np.isfinite(v) or v < 0 for v in grid):
            raise ValueError("shrink_grid must contain finite nonnegative values")
        fid = make_folds(groups, n_folds, seed)
        for candidate in grid:
            pred = np.empty_like(y)
            for fold in np.unique(fid):
                train, valid = fid != fold, fid == fold
                model = Calibration().fit(x[train], y[train], codes[train], groups[train], mode, candidate)
                pred[valid] = model.predict(x[valid], codes[valid])
            audit["cv"].append({"shrink": candidate, "mse": group_mse(pred, y, groups)})
        # At a tie prefer stronger shrinkage, the simpler diagonal correction.
        best = min(audit["cv"], key=lambda row: (row["mse"], -row["shrink"]))
        chosen = best["shrink"]
        audit.update(selection="FIT grouped CV", actual_folds=int(fid.max()) + 1)
    fitted = Calibration().fit(x, y, codes, groups, mode, chosen)
    audit["selected_shrink"] = chosen
    return fitted, audit


def crossfit_calibration(x, y, codes, groups, mode="scalar", shrink=None, n_folds=8,
                         seed=DEFAULT_SEED, shrink_grid=DEFAULT_SHRINK_GRID):
    """OOF baselines with diagonal hyperparameter selection nested inside each fold."""
    x, y, codes, groups = _calibration_arrays(x, y, codes, groups)
    fid = make_folds(groups, n_folds, seed)
    prediction, audits = np.empty_like(y), []
    for fold in np.unique(fid):
        train, valid = fid != fold, fid == fold
        model, audit = fit_calibration(x[train], y[train], codes[train], groups[train],
            mode=mode, shrink=shrink, n_folds=n_folds, seed=seed + 1009 + int(fold), shrink_grid=shrink_grid)
        prediction[valid] = model.predict(x[valid], codes[valid])
        audits.append({"fold": int(fold), "validation_groups": np.unique(groups[valid]).tolist(),
                       "baseline": audit})
    return prediction, {"folds": audits, "actual_folds": int(fid.max()) + 1,
                        "all_rows_predicted_once": True, "diagonal_selection_nested": True}


@dataclass
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X, groups):
        X = _matrix(X, "X")
        w = group_weights(_labels(groups, "groups", len(X)))
        mean = w @ X
        scale = np.sqrt(w @ ((X - mean) ** 2))
        scale[scale < 1e-8] = 1.
        return cls(mean, scale)

    def transform(self, X):
        X = _matrix(X, "X")
        if X.shape[1] != len(self.mean):
            raise ValueError("Ridge input dimension does not match")
        return (X - self.mean) / self.scale

    def to_dict(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, value):
        mean, scale = np.asarray(value["mean"], dtype=float), np.asarray(value["scale"], dtype=float)
        if mean.ndim != 1 or not len(mean) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid saved ridge standardizer")
        return cls(mean, scale)


def _indices(value, rows, name):
    value = np.asarray(value)
    if value.dtype == bool:
        if value.shape != (rows,):
            raise ValueError(f"{name} mask has incorrect length")
        value = np.flatnonzero(value)
    if value.ndim != 1 or not len(value) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must contain row indices")
    if np.any(value < 0) or np.any(value >= rows) or len(np.unique(value)) != len(value):
        raise ValueError(f"Invalid {name}")
    return value.astype(np.int64)


def prepare_inner_split(x, y, X, codes, groups, train_indices, valid_indices, *,
                        mode="scalar", shrink=None, n_folds=8, seed=DEFAULT_SEED,
                        shrink_grid=DEFAULT_SHRINK_GRID):
    """Build an outer-fold training pipeline without reading its validation targets.

    Exposed for isolation tests. Returned residuals and preprocessing are fitted
    only from ``train_indices``; validation inputs are used only for prediction.
    """
    x = _matrix(x, "x")
    y = np.asarray(y, dtype=np.float64)
    if y.shape != x.shape:
        raise ValueError("Copy calibration requires matching source and output features")
    codes, groups = _labels(codes, "codes", len(x)), _labels(groups, "groups", len(x))
    X = _matrix(X, "X", len(x))
    train = _indices(train_indices, len(x), "train_indices")
    valid = _indices(valid_indices, len(x), "valid_indices")
    if set(groups[train]) & set(groups[valid]):
        raise ValueError("Training and validation chemical groups overlap")
    # Only training targets are validated/read here; held-fold targets may even
    # be unavailable (NaN) while this preparation step is performed.
    _matrix(y[train], "training targets")
    baseline, baseline_audit = fit_calibration(x[train], y[train], codes[train], groups[train],
        mode=mode, shrink=shrink, n_folds=n_folds, seed=seed, shrink_grid=shrink_grid)
    oof, oof_audit = crossfit_calibration(x[train], y[train], codes[train], groups[train],
        mode=mode, shrink=shrink, n_folds=n_folds, seed=seed + 7919, shrink_grid=shrink_grid)
    scaler = _Standardizer.fit(X[train], groups[train])
    return {"base": baseline, "scaler": scaler,
            "residual_train": y[train] - oof,
            "X_train": scaler.transform(X[train]), "X_valid": scaler.transform(X[valid]),
            "base_valid": baseline.predict(x[valid], codes[valid]),
            "train_indices": train, "valid_indices": valid,
            "audit": {"train_groups": np.unique(groups[train]).tolist(),
                      "validation_groups": np.unique(groups[valid]).tolist(),
                      "baseline": baseline_audit, "residual_crossfit": oof_audit,
                      "validation_targets_used_upstream": False}}


class _RidgePath:
    """Dual eigendecomposition reused over a fixed positive-lambda grid."""
    def __init__(self, X, residual, groups):
        self.X = X
        w = group_weights(groups)
        self.mean = w @ residual
        self.sqrt_weight = np.sqrt(len(X) * w)
        self.Z = self.sqrt_weight[:, None] * X
        kernel = self.Z @ self.Z.T
        values, vectors = np.linalg.eigh((kernel + kernel.T) * .5)
        self.values, self.vectors = np.maximum(values, 0.), vectors
        self.projected_y = vectors.T @ (self.sqrt_weight[:, None] * (residual - self.mean))

    def coefficients(self, lam):
        return self.Z.T @ (self.vectors @ (self.projected_y / (self.values[:, None] + lam)))

    def predict(self, X, lam):
        return ((X @ self.Z.T) @ self.vectors) @ (self.projected_y / (self.values[:, None] + lam)) + self.mean


def _basis(value, dimension):
    if value is None:
        return None
    value = _matrix(value, "output_basis", dimension)
    if value.shape[1] > dimension or not np.allclose(value.T @ value, np.eye(value.shape[1]), atol=1e-7, rtol=1e-7):
        raise ValueError("output_basis must have orthonormal columns")
    return value.copy()


@dataclass
class ResidualRidge:
    base: Calibration
    scaler: _Standardizer
    coefficients: np.ndarray
    residual_mean: np.ndarray
    output_basis: np.ndarray | None
    selected_lambda: float | None
    audit: dict[str, Any]

    def predict(self, x, X, codes):
        base = self.base.predict(x, codes)
        X = self.scaler.transform(X)
        if len(X) != len(base):
            raise ValueError("Source and ridge inputs have different row counts")
        if self.selected_lambda is None:
            return base
        correction = X @ self.coefficients + self.residual_mean
        if self.output_basis is not None:
            correction = correction @ self.output_basis.T
        return base + correction

    def to_dict(self):
        return {"format": "v7_residual_ridge", "base": self.base.to_dict(),
                "scaler": self.scaler.to_dict(), "coefficients": self.coefficients.tolist(),
                "residual_mean": self.residual_mean.tolist(),
                "output_basis": None if self.output_basis is None else self.output_basis.tolist(),
                "selected_lambda": self.selected_lambda, "audit": self.audit}

    @classmethod
    def from_dict(cls, value):
        if value.get("format") != "v7_residual_ridge":
            raise ValueError("Unknown residual ridge format")
        base, scaler = Calibration.from_dict(value["base"]), _Standardizer.from_dict(value["scaler"])
        basis = _basis(value["output_basis"], base.n_features)
        coefficients = _matrix(value["coefficients"], "coefficients", len(scaler.mean))
        dimension = base.n_features if basis is None else basis.shape[1]
        mean = np.asarray(value["residual_mean"], dtype=float)
        if coefficients.shape[1] != dimension or mean.shape != (dimension,) or not np.isfinite(mean).all():
            raise ValueError("Invalid saved ridge output dimensions")
        lam = value["selected_lambda"]
        if lam is not None:
            lam = float(lam)
            if not np.isfinite(lam) or lam <= 0:
                raise ValueError("Invalid saved ridge lambda")
        return cls(base, scaler, coefficients, mean, basis, lam, dict(value["audit"]))


def fit_residual_ridge(x, y, X, codes, groups, output_basis=None, seed=DEFAULT_SEED,
                       lambdas=DEFAULT_LAMBDAS, *, mode="scalar", shrink=None,
                       n_folds=8, inner_folds=None, shrink_grid=DEFAULT_SHRINK_GRID):
    """Select a residual ridge by genuinely nested chemical CV, then refit on FIT.

    ``None``/positive infinity denote an exact no-correction candidate, which is
    always included. A supplied output basis must be fixed independently of these
    targets; a data-derived basis needs its own foldwise fitting and must not be
    passed here as a fixed matrix. All loss and reconstruction are in raw units.
    """
    x, y, codes, groups = _calibration_arrays(x, y, codes, groups)
    X = _matrix(X, "X", len(x))
    basis = _basis(output_basis, y.shape[1])
    inner_folds = n_folds if inner_folds is None else inner_folds
    candidates = [None]
    for value in lambdas:
        if value is None or float(value) == float("inf"):
            continue
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Finite ridge lambdas must be positive")
        if value not in candidates:
            candidates.append(value)
    fid = make_folds(groups, n_folds, seed)
    curve = np.zeros(len(candidates), dtype=float)
    outer_audits = []
    total_groups = len(np.unique(groups))
    for fold in np.unique(fid):
        train, valid = np.flatnonzero(fid != fold), np.flatnonzero(fid == fold)
        prepared = prepare_inner_split(x, y, X, codes, groups, train, valid,
            mode=mode, shrink=shrink, n_folds=inner_folds,
            seed=seed + 104729 * (int(fold) + 1), shrink_grid=shrink_grid)
        residual = prepared["residual_train"]
        if basis is not None:
            residual = residual @ basis
        path = _RidgePath(prepared["X_train"], residual, groups[train])
        fold_weight = len(np.unique(groups[valid])) / total_groups
        fold_curve = []
        for index, lam in enumerate(candidates):
            prediction = prepared["base_valid"]
            if lam is not None:
                correction = path.predict(prepared["X_valid"], lam)
                if basis is not None:
                    correction = correction @ basis.T
                prediction = prediction + correction
            risk = group_mse(prediction, y[valid], groups[valid])
            curve[index] += fold_weight * risk
            fold_curve.append({"lambda": lam, "mse": risk})
        outer_audits.append({"fold": int(fold), **prepared["audit"], "cv": fold_curve})
    # Exact ties prefer the zero correction, then the strongest finite shrinkage.
    best = min(range(len(candidates)), key=lambda i: (curve[i], 0 if candidates[i] is None else 1,
                                                    -(candidates[i] or 0.)))
    selected = candidates[best]
    base, base_audit = fit_calibration(x, y, codes, groups, mode=mode, shrink=shrink,
        n_folds=inner_folds, seed=seed + 200003, shrink_grid=shrink_grid)
    oof, oof_audit = crossfit_calibration(x, y, codes, groups, mode=mode, shrink=shrink,
        n_folds=inner_folds, seed=seed + 200009, shrink_grid=shrink_grid)
    residual = y - oof
    if basis is not None:
        residual = residual @ basis
    scaler = _Standardizer.fit(X, groups)
    output_dimension = residual.shape[1]
    if selected is None:
        coefficients, residual_mean = np.zeros((X.shape[1], output_dimension)), np.zeros(output_dimension)
    else:
        path = _RidgePath(scaler.transform(X), residual, groups)
        coefficients, residual_mean = path.coefficients(selected), path.mean
    audit = {"method": "nested grouped residual ridge", "fit_rows": len(x),
             "fit_groups": np.unique(groups).tolist(), "seed": int(seed),
             "actual_outer_folds": int(fid.max()) + 1, "outer_folds": outer_audits,
             "cv": [{"lambda": lam, "mse": float(risk)} for lam, risk in zip(candidates, curve)],
             "selected_lambda": selected, "zero_correction_included": True,
             "baseline": base_audit, "final_residual_crossfit": oof_audit,
             "loss": "chemical-equal raw-coordinate MSE",
             "ridge_penalty_convention": "sum_i n*w_i ||e_i||^2 + lambda*||W||_F^2; sum(w)=1",
             "output_basis": "full output" if basis is None else "caller-supplied fixed independent orthonormal basis",
             "validation_targets_used_upstream": False}
    return ResidualRidge(base, scaler, coefficients, residual_mean, basis, selected, audit)
