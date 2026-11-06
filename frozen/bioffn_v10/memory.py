"""Domain-standardized coordinates and weighted cross-resource residual memory.

The caller fixes G, coordinate maps, kernel geometry, local intercept and
reference row count before changing memory size. For row weights W, this module
solves (sqrt(W) K sqrt(W) + lambda I) b = sqrt(W) R and alpha=sqrt(W)b.
Local targets are R_local minus the fixed local mean; source targets must
already be centered on the source representation-training mean. The query
intercept is the local mean only. No dataset is loaded or split here.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from bioffn_v7 import baselines as BL
from bioffn_v8 import memory as V8
from bioffn_v9 import memory as V9


@dataclass
class DomainCoordinates:
    coordinates: V9.MemoryCoordinates
    audit: dict[str, Any]

    @classmethod
    def fit(cls, x, rna, conditions, groups, provider=None, rank=16):
        """Fit all axes/scales on local training inputs, without any outcomes."""
        coordinates = V9.MemoryCoordinates.fit(x, rna, conditions, groups,
            key_predictor=provider, x_rank=rank, rna_rank=rank)
        audit = {"local_coordinates": copy.deepcopy(coordinates.audit),
            "outcomes_passed_to_coordinate_fit": False, "provider_may_be_supervised": True,
            "source_rows_used_to_fit_axes": False,
            "local_transform": "V9 destination-training moments and PCA",
            "source_x_transform": "source representation-training mean/SD, then destination x PCA and block RMS",
            "source_rna_transform": "caller-supplied source-training gene-standardized RNA; not destination restandardized",
            "source_key_transform": "provider(source_z), then destination response-column mean/SD, PCA and block RMS; raw mode uses destination RNA PCA",
            "condition_transform": "destination condition-column mean/SD for both domains"}
        return cls(coordinates, audit)

    def transform_local(self, x, rna, conditions):
        return self.coordinates.transform(x, rna, conditions)

    def transform_query(self, x, rna, conditions):
        return self.transform_local(x, rna, conditions)

    def transform_source(self, x, source_z, conditions, x_mean_source, x_scale_source):
        """Map source rows without fitting moments or accessing source outcomes."""
        x = BL._matrix(x, "source x")
        z = BL._matrix(source_z, "source-standardized RNA", len(x))
        c = BL._matrix(conditions, "source conditions", len(x))
        common = self.coordinates.common
        xm, xs = np.asarray(x_mean_source, float), np.asarray(x_scale_source, float)
        if (x.shape[1] != len(common.x_mean) or xm.shape != common.x_mean.shape
                or xs.shape != xm.shape or not np.isfinite(xm).all()
                or not np.isfinite(xs).all() or np.any(xs <= 0)):
            raise ValueError("Source morphology moments must be finite, positive-scale and aligned")
        if z.shape[1] != len(common.rna_mean) or c.shape[1] != len(common.condition_mean):
            raise ValueError("Source RNA/condition dimensions differ from destination coordinates")
        blocks = [((x - xm) / xs) @ common.x_components / common.x_block_scale]
        coord = self.coordinates
        if coord.key_mean is None:
            key = z @ common.rna_components / common.rna_block_scale
        else:
            response = BL._matrix(coord.key_predictor.predict_standardized(z),
                                  "source response key", len(x))
            if response.shape[1] != len(coord.key_mean):
                raise ValueError("Source response key dimension differs from fitted coordinates")
            key = ((response - coord.key_mean) / coord.key_scale) @ coord.key_components / coord.key_block_scale
        blocks.extend((key, (c - common.condition_mean) / common.condition_scale))
        return np.concatenate(blocks, axis=1)

    def to_dict(self):
        return {"format": "v10_domain_coordinates", "coordinates": self.coordinates.to_dict(),
                "audit": copy.deepcopy(self.audit)}

    @classmethod
    def from_dict(cls, saved, provider=None):
        if saved.get("format") != "v10_domain_coordinates":
            raise ValueError("Unknown domain coordinate format")
        return cls(V9.MemoryCoordinates.from_dict(saved["coordinates"], key_predictor=provider),
                   copy.deepcopy(saved["audit"]))


@dataclass
class WeightedMemoryModel:
    train_features: np.ndarray
    dual_coefficients: np.ndarray
    intercept: np.ndarray
    geometry: dict[str, Any]
    weight: float | str
    bandwidth: float
    audit: dict[str, Any]

    def predict(self, features):
        features = BL._matrix(features, "query features")
        return self.intercept + V8.kernel_matrix(self.train_features, features,
            self.weight, self.bandwidth, self.geometry) @ self.dual_coefficients

    def to_dict(self):
        return {"format": "v10_weighted_memory", "train_features": self.train_features.tolist(),
            "dual_coefficients": self.dual_coefficients.tolist(), "intercept": self.intercept.tolist(),
            "geometry": copy.deepcopy(self.geometry), "weight": self.weight,
            "bandwidth": self.bandwidth, "audit": copy.deepcopy(self.audit)}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v10_weighted_memory":
            raise ValueError("Unknown weighted memory format")
        features = BL._matrix(saved["train_features"], "saved features").copy()
        alpha = BL._matrix(saved["dual_coefficients"], "saved dual coefficients", len(features)).copy()
        intercept = np.asarray(saved["intercept"], float)
        if intercept.shape != (alpha.shape[1],) or not np.isfinite(intercept).all():
            raise ValueError("Invalid saved intercept")
        geometry = _geometry(saved["geometry"])
        _kernel_parameters(saved["weight"], saved["bandwidth"])
        return cls(features, alpha, intercept.copy(), geometry, saved["weight"],
                   float(saved["bandwidth"]), copy.deepcopy(saved["audit"]))


def _kernel_parameters(weight, bandwidth):
    if weight != "rbf" and (not isinstance(weight, (int, float, np.number))
                            or not np.isfinite(weight) or weight < 0):
        raise ValueError("Kernel weight must be nonnegative or 'rbf'")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be finite and positive")


def _geometry(value):
    result = copy.deepcopy(value)
    if not isinstance(result, dict):
        raise ValueError("Kernel geometry must be a dictionary")
    for key in ("distance_reference", "linear_scale"):
        if key not in result or not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError("Kernel geometry scales must be finite and positive")
    return result


def fit_weighted_memory(local_features, local_residual, local_groups,
                        source_features, source_residual, source_groups, rho, *,
                        lambda_value=100., weight="rbf", bandwidth=1., geometry=None,
                        local_reference_rows=None, local_residual_mean=None):
    """Fit full-output KRR, preserving local/source chemical-equal mass.

    Objective data weights sum to n_ref locally and n_ref*rho externally.
    Source targets are accepted already centered, with NO second centering.
    At rho=0 source arguments are not read, validated, or included in the Gram
    matrix. Explicit fixed geometry/mean/n_ref are required by the caller for
    a controlled memory-size curve; defaults describe a single full-local fit.
    """
    local = BL._matrix(local_features, "local features")
    residual = BL._matrix(local_residual, "local residual", len(local))
    groups = BL._labels(local_groups, "local groups", len(local))
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("External mass rho must be finite and nonnegative")
    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("Ridge lambda must be finite and positive")
    _kernel_parameters(weight, bandwidth)
    n_ref = len(local) if local_reference_rows is None else local_reference_rows
    if (isinstance(n_ref, (bool, np.bool_)) or not isinstance(n_ref, (int, np.integer))
            or n_ref < len(local)):
        raise ValueError("Reference row count must be an integer at least the local subset size")
    w_local = BL.group_weights(groups)
    mean = w_local @ residual if local_residual_mean is None else np.asarray(local_residual_mean, float)
    if mean.shape != (residual.shape[1],) or not np.isfinite(mean).all():
        raise ValueError("Local residual mean must match the full output dimension")
    geo = _geometry(V8.kernel_geometry(local) if geometry is None else geometry)
    data_weight = n_ref*w_local
    features, targets = local, residual-mean
    n_source, source_chemicals, w_source = 0, 0, np.array([], dtype=float)
    if rho > 0:
        source = BL._matrix(source_features, "source features")
        source_target = BL._matrix(source_residual, "centered source residual", len(source))
        source_group = BL._labels(source_groups, "source groups", len(source))
        if source.shape[1] != local.shape[1] or source_target.shape[1] != residual.shape[1]:
            raise ValueError("Source/local feature and full-output dimensions must align")
        n_source, source_chemicals = len(source), len(np.unique(source_group))
        w_source = n_ref*float(rho)*BL.group_weights(source_group)
        data_weight = np.concatenate((data_weight, w_source))
        features = np.concatenate((local, source))
        targets = np.concatenate((targets, source_target))
    sqrt_weight = np.sqrt(data_weight)
    kernel = V8.kernel_matrix(features, features, weight, bandwidth, geo)
    system = sqrt_weight[:, None]*kernel*sqrt_weight[None, :]
    system = (system+system.T)*.5
    system.flat[::len(system)+1] += lambda_value
    factor = cho_factor(system, lower=True, check_finite=True)
    scaled_alpha = cho_solve(factor, sqrt_weight[:, None]*targets, check_finite=True)
    alpha = np.ascontiguousarray(sqrt_weight[:, None]*scaled_alpha)
    normal_error = system@scaled_alpha - sqrt_weight[:, None]*targets
    denominator = max(1., float(np.linalg.norm(sqrt_weight[:, None]*targets)))
    audit = {"method": "weighted local/external full-output kernel residual memory",
        "rho": float(rho), "lambda": float(lambda_value), "weight": weight,
        "bandwidth": float(bandwidth), "local_rows": len(local), "source_rows_used": n_source,
        "local_chemicals": len(np.unique(groups)), "source_chemicals": source_chemicals,
        "local_reference_rows": int(n_ref), "local_weight_sum": float(n_ref*w_local.sum()),
        "source_weight_sum": float(w_source.sum()), "local_row_weights": (n_ref*w_local).tolist(),
        "source_row_weights": w_source.tolist(), "geometry": geo,
        "reference_rows_explicit": local_reference_rows is not None,
        "geometry_explicit": geometry is not None, "local_intercept_explicit": local_residual_mean is not None,
        "intercept_scope": "fixed full-local residual mean only",
        "source_centering": "caller supplied source representation-training centered targets; not re-centered here",
        "source_arguments_ignored": rho == 0, "full_output_dim": residual.shape[1],
        "objective": "n_ref sum local_group_weights ||R_local-mean-f||² + n_ref*rho sum source_group_weights ||R_source_centered-f||² + lambda ||f||_H²",
        "solver": "symmetric weighted Cholesky; no added jitter",
        "relative_normal_equation_error": float(np.linalg.norm(normal_error)/denominator)}
    return WeightedMemoryModel(features.copy(), alpha, mean.copy(), geo, weight,
                               float(bandwidth), audit)
