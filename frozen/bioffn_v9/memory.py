"""V9 separates the external prior mean from the local kernel's RNA keys.

Raw keys reproduce V8. An external key provider sees destination-training
gene-standardized RNA; its outputs receive fold-local column scaling, PCA,
and one block RMS. The full response is still predicted in original units.
No data are loaded, external providers fitted, or module globals changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np

from bioffn_v7 import baselines as BL
from bioffn_v7.features import _pca
from bioffn_v8 import memory as V8


def _provider_output(provider, coordinates, rna):
    z = coordinates.standardize_rna(rna)
    return BL._matrix(provider.predict_standardized(z), "external key output", len(rna))


@dataclass
class MemoryCoordinates:
    common: V8.MemoryCoordinates
    key_mean: np.ndarray | None
    key_scale: np.ndarray | None
    key_components: np.ndarray | None
    key_block_scale: float | None
    key_predictor: Any
    audit: dict[str, Any]

    @classmethod
    def fit(cls, x, rna, conditions, groups, *, key_predictor=None,
            x_rank=16, rna_rank=16, include_rna=True):
        # In raw mode use exactly the V8 fit and transform, including operation
        # order. Its RNA moments also define both external predictor interfaces.
        common = V8.MemoryCoordinates.fit(x, rna, conditions, groups,
            x_rank=x_rank, rna_rank=rna_rank, include_rna=include_rna)
        km = ks = kp = kb = None
        if key_predictor is not None and include_rna:
            output = _provider_output(key_predictor, common, rna)
            km, ks, kp, kb, ka = _pca(output, BL.group_weights(groups), rna_rank)
            ka = dict(ka, input_dim=output.shape[1], transform="external response or fixed projection",
                extra_scaling="destination TRAIN per-output-column mean/SD BEFORE PCA; then one block RMS",
                provider_class=type(key_predictor).__name__,
                provider_kind=getattr(key_predictor, "kind", None), provider_refitted=False)
        else:
            ka = dict(common.audit["rna_pca"], input_dim=np.shape(rna)[1],
                transform="raw RNA" if include_rna else "RNA omitted",
                extra_scaling="V8 destination TRAIN gene mean/SD; PCA; one block RMS",
                provider_refitted=False)
        ka.update(fit_groups=np.unique(groups).astype(str).tolist(),
                  include_key_block=bool(include_rna),
                  coordinates_fixed_across_local_hyperparameters=True,
                  prior_mean_not_used_in_key_map=True)
        audit = dict(common.audit, key_map=ka,
                     key_mode="external" if key_predictor is not None and include_rna else "raw",
                     key_and_mean_predictors_separate=True)
        return cls(common, km, ks, kp, kb, key_predictor, audit)

    def standardize_rna(self, rna):
        return self.common.standardize_rna(rna)

    def transform(self, x, rna, conditions):
        if self.key_mean is None:
            return self.common.transform(x, rna, conditions)
        # Run common validation without borrowing the common RNA PCA scores.
        common_features = self.common.transform(x, rna, conditions)
        output = _provider_output(self.key_predictor, self.common, rna)
        if output.shape[1] != len(self.key_mean):
            raise ValueError("External key output dimension changed")
        key = ((output - self.key_mean) / self.key_scale) @ self.key_components / self.key_block_scale
        xdim = self.common.x_components.shape[1]
        cdim = len(self.common.condition_mean)
        return np.concatenate((common_features[:, :xdim], key, common_features[:, -cdim:]), axis=1)

    def to_dict(self):
        return {"format": "v9_memory_coordinates", "common": self.common.to_dict(),
                "key_mean": None if self.key_mean is None else self.key_mean.tolist(),
                "key_scale": None if self.key_scale is None else self.key_scale.tolist(),
                "key_components": None if self.key_components is None else self.key_components.tolist(),
                "key_block_scale": self.key_block_scale,
                "key_provider_required": self.key_mean is not None,
                "audit": copy.deepcopy(self.audit)}

    @classmethod
    def from_dict(cls, value, key_predictor=None):
        if value.get("format") != "v9_memory_coordinates":
            raise ValueError("Unknown V9 coordinate format")
        required = bool(value["key_provider_required"])
        if required and key_predictor is None:
            raise ValueError("Saved external coordinates require the key predictor")
        common = V8.MemoryCoordinates.from_dict(value["common"])
        km = ks = kp = kb = None
        if required:
            km, ks, kp = (np.asarray(value[k], float) for k in ("key_mean", "key_scale", "key_components"))
            kb = float(value["key_block_scale"])
            if (km.ndim != 1 or ks.shape != km.shape or kp.ndim != 2 or kp.shape[0] != len(km)
                    or not all(np.isfinite(v).all() for v in (km, ks, kp))
                    or np.any(ks <= 0) or not np.isfinite(kb) or kb <= 0):
                raise ValueError("Invalid saved key map")
        return cls(common, km, ks, kp, kb, key_predictor, copy.deepcopy(value["audit"]))


def prepare_memory_split(x, y, rna, conditions, codes, groups, train_indices,
                         valid_indices, prior_predictor=None, *, key_predictor=None,
                         folds=8, seed=20261008, x_rank=16, rna_rank=16, include_rna=True):
    """Build features and honest residuals without any validation target use."""
    x, rna, conditions, codes, groups = V8._inputs(x, rna, conditions, codes, groups)
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
        key_predictor=key_predictor, x_rank=x_rank, rna_rank=rna_rank, include_rna=include_rna)
    return {"base": base, "coordinates": coords, "residual_train": y[train] - oof,
            "X_train": coords.transform(x[train], rna[train], conditions[train]),
            "X_valid": coords.transform(x[valid], rna[valid], conditions[valid]),
            "base_valid": base.predict(x[valid], codes[valid]),
            "prior_train": V8._prior(prior_predictor, coords, rna[train], y.shape[1]),
            "prior_valid": V8._prior(prior_predictor, coords, rna[valid], y.shape[1]),
            "audit": {"train_indices": train.tolist(), "validation_indices": valid.tolist(),
                      "train_groups": np.unique(groups[train]).tolist(),
                      "validation_groups": np.unique(groups[valid]).tolist(),
                      "baseline": ba, "residual_crossfit": oa, "coordinates": coords.audit,
                      "key_initial": copy.deepcopy(coords.audit["key_map"]),
                      "baseline_target": "original y", "validation_targets_used_upstream": False}}


@dataclass
class MemoryModel(V8.MemoryModel):
    cv_prediction: np.ndarray

    def to_dict(self):
        value = super().to_dict()
        value.update(format="v9_memory_model", cv_prediction=self.cv_prediction.tolist())
        return value

    @classmethod
    def from_dict(cls, value, prior_predictor=None, key_predictor=None):
        if value.get("format") != "v9_memory_model":
            raise ValueError("Unknown V9 memory format")
        if value["prior_required"] and prior_predictor is None:
            raise ValueError("Saved model requires its external prior predictor")
        base = BL.Calibration.from_dict(value["base"])
        coords = MemoryCoordinates.from_dict(value["coordinates"], key_predictor)
        features = BL._matrix(value["train_features"], "train_features")
        alpha = BL._matrix(value["dual_coefficients"], "dual_coefficients", len(features))
        mean = np.asarray(value["residual_mean"], float)
        cv = BL._matrix(value["cv_prediction"], "cv_prediction", len(features))
        if alpha.shape[1] != base.n_features or cv.shape != alpha.shape or mean.shape != (base.n_features,):
            raise ValueError("Saved output dimensions disagree")
        if not np.isfinite(mean).all():
            raise ValueError("Nonfinite residual mean")
        return cls(base, coords, features, alpha, mean, copy.deepcopy(value["selected"]),
            dict(value["geometry"]), prior_predictor, copy.deepcopy(value["audit"]), cv)


def fit_memory(x, y, rna, conditions, codes, groups, prior_predictor=None, *,
               key_predictor=None, kind="kernel", seed=20261008, folds=8,
               scale_grid=(0., .5, 1.), include_rna=True, x_rank=16, rna_rank=16,
               lambdas=BL.DEFAULT_LAMBDAS, kernel_weights=V8.DEFAULT_WEIGHTS,
               bandwidths=V8.DEFAULT_BANDWIDTHS):
    """Select mean strength and ordinary KRR independently of the fixed key map.

    ``cv_prediction`` is aligned to input FIT rows and uses each row's outer
    training split. It is selection-biased because those folds selected the
    configuration; it is not an additional held-out performance estimate.
    """
    x, rna, conditions, codes, groups = V8._inputs(x, rna, conditions, codes, groups)
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
    candidates = [V8._candidate(s) for s in scales]
    candidates += [V8._candidate(s, True, w, b, lam) for w, b in kernels for lam in lams for s in scales]
    curve = np.zeros(len(candidates))
    fid = BL.make_folds(groups, folds, seed)
    audits, cache = [], []
    total_groups = len(np.unique(groups))
    for fold in np.unique(fid):
        train, valid = np.flatnonzero(fid != fold), np.flatnonzero(fid == fold)
        p = prepare_memory_split(x, y, rna, conditions, codes, groups, train, valid,
            prior_predictor, key_predictor=key_predictor, folds=folds,
            seed=seed + 104729 * (int(fold) + 1), x_rank=x_rank, rna_rank=rna_rank,
            include_rna=include_rna)
        fw = len(np.unique(groups[valid])) / total_groups
        tw, vw = BL.group_weights(groups[train]), BL.group_weights(groups[valid])
        stats, risks = {}, np.empty(len(candidates))
        for i, scale in enumerate(scales):
            pred = p["base_valid"] + scale * p["prior_valid"]
            risks[i] = BL.group_mse(pred, y[valid], groups[valid])
            stats[scale] = V8._risk_stats(p["residual_train"] - scale*p["prior_train"],
                                         y[valid] - pred, tw, vw)
        cursor, geometry = len(scales), V8.kernel_geometry(p["X_train"])
        for weight, band in kernels:
            path = V8._KernelPath(p["X_train"], groups[train], weight, band, geometry)
            for lam in lams:
                transfer = path.transfer(p["X_valid"], lam)
                for scale in scales:
                    risks[cursor] = V8._risk_from_transfer(transfer, stats[scale])
                    cursor += 1
        curve += fw*risks
        audits.append({"fold": int(fold), **p["audit"], "kernel_geometry": geometry,
                       "candidate_mse": risks.tolist()})
        # Retain one preparation per fold, not a grid of full-output predictions.
        cache.append((train, valid, p, geometry))
    best = V8._choose_index(curve, candidates)
    selected = copy.deepcopy(candidates[best])
    cv = np.empty_like(y)
    for audit, (train, valid, p, geo) in zip(audits, cache):
        pred = p["base_valid"] + selected["scale"]*p["prior_valid"]
        if selected["memory"]:
            path = V8._KernelPath(p["X_train"], groups[train], selected["weight"], selected["bandwidth"], geo)
            a, m = path.solve(p["residual_train"]-selected["scale"]*p["prior_train"], selected["lambda"])
            pred = pred + V8.kernel_matrix(p["X_train"], p["X_valid"], selected["weight"],
                                          selected["bandwidth"], geo) @ a + m
        cv[valid] = pred
        audit["selected_cv_mse"] = BL.group_mse(pred, y[valid], groups[valid])
        audit["key_selected"] = copy.deepcopy(audit["key_initial"])
    del cache
    selected_risk = BL.group_mse(cv, y, groups)
    if not np.isclose(selected_risk, curve[best], atol=1e-8, rtol=1e-10):
        raise ArithmeticError("Selected full-output CV predictions disagree with selection risk")
    base, ba = BL.fit_calibration(x, y, codes, groups, mode="scalar", n_folds=folds, seed=seed+200003)
    oof, oa = BL.crossfit_calibration(x, y, codes, groups, mode="scalar", n_folds=folds, seed=seed+200009)
    coords = MemoryCoordinates.fit(x, rna, conditions, groups, key_predictor=key_predictor,
        x_rank=x_rank, rna_rank=rna_rank, include_rna=include_rna)
    features = coords.transform(x, rna, conditions)
    geometry = V8.kernel_geometry(features)
    if selected["memory"]:
        residual = y-oof-selected["scale"]*V8._prior(prior_predictor, coords, rna, y.shape[1])
        path = V8._KernelPath(features, groups, selected["weight"], selected["bandwidth"], geometry)
        alpha, mean = path.solve(residual, selected["lambda"])
    else:
        alpha, mean = np.zeros_like(y), np.zeros(y.shape[1])
    audit = {"method": "v9 independent external key / prior residual memory", "kind": kind,
             "fit_rows": len(x), "fit_groups": np.unique(groups).tolist(), "seed": int(seed),
             "actual_outer_folds": len(audits), "outer_folds": audits,
             "candidates": [{**c, "mse": float(v)} for c, v in zip(candidates, curve)],
             "selected": selected, "selected_cv_mse": float(curve[best]),
             "baseline": ba, "final_residual_crossfit": oa, "coordinates": coords.audit,
             "baseline_target": "original y; identical across key and prior choices",
             "residual_target": "y - group-OOF G(original y) - s * external mean prior",
             "key_mode": coords.audit["key_mode"], "key_mean_coupling": False,
             "key_predictor_present": key_predictor is not None,
             "prior_present": prior_predictor is not None,
             "loss": "chemical-equal original-coordinate MSE, all output features",
             "ridge_penalty": "sum_i n*w_i ||e_i||^2 + lambda ||f||_H^2, sum(w)=1",
             "intercept": "training residual mean after subtracting the mean prior",
             "selection_scope": "FIT nested grouped CV only; no TUNE/evaluation targets",
             "zero_baseline_candidate_included": True, "full_output": True,
             "include_rna_in_memory": bool(include_rna),
             "cv_prediction": {"row_order": "input FIT order", "all_rows_predicted_once": True,
                "selected_candidate": selected, "selection_biased": True,
                "interpretation": "fold-held predictions of CV-selected candidate; not independent validation"},
             "provider_fit_scope": "external frozen callers; never fitted by local memory"}
    return MemoryModel(base, coords, features, alpha, mean, selected, geometry, prior_predictor, audit, cv)
