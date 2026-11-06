"""Frozen V15 memory and the two declared V16 secondary estimators.

All inputs are already frozen 39-column coordinates: morphology (16), atlas
RNA RESPONSE key (16), condition (7). This module loads no project data and
does not fit G or coordinate maps. Every prediction is a residual correction;
the caller adds the same frozen G prediction.

The main arm calls the exact V15 estimator (V14 AND memory with phi=1,
bandwidth=1, lambda=100). Angular memory changes only the response-key block.
Its source-FIT chemical-equal center and positive off-diagonal median
``||unit_key_i - unit_key_j||²/2`` are frozen before references are inserted.
Zero centered vectors map to zero. The kernel uses squared Euclidean distance
between these unit-or-zero vectors divided by 2h: this equals (1-cos)/h for
nonzero pairs and remains PSD when zero vectors occur. A zero/nonzero pair
therefore has distance 1/2, and a zero/zero pair has distance 0, before /h.

The shift arm removes source and target-reference condition means separately,
fits their deviations using the same n*chemical_group_weights and fixed RBF,
and adds only the target-reference condition mean at prediction. With no
references at all it exactly falls back to the V15 source-only arm. A condition
absent from a nonempty reference set uses its source mean as the fixed shift.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from bioffn_v7 import baselines as BL
from bioffn_v8.memory import kernel_matrix as _rbf
from bioffn_v10.memory import _geometry
from bioffn_v13.kernel import RNA_SLICE, _features, _squared_distance
from bioffn_v14.memory import MemoryModel as V15MemoryModel
from bioffn_v14.memory import fit_memory as _fit_v15


LAMBDA = 100.
BANDWIDTH = 1.
VARIANTS = ("main", "angular", "shift")
_EPS = 1e-12


def _unit_rows(values):
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("Response-key magnitude overflowed")
    return np.divide(values, norms[:, None], out=np.zeros_like(values),
                     where=norms[:, None] > _EPS)


@dataclass
class AngularGeometry:
    response_center: np.ndarray
    bandwidth: float
    audit: dict

    def to_dict(self):
        return dict(format="v16_angular_geometry", response_center=self.response_center.tolist(),
                    bandwidth=self.bandwidth, audit=copy.deepcopy(self.audit))

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v16_angular_geometry":
            raise ValueError("Unknown V16 angular geometry format")
        center = np.asarray(saved["response_center"], dtype=float)
        h = float(saved["bandwidth"])
        if center.shape != (16,) or not np.isfinite(center).all() or not np.isfinite(h) or h <= 0:
            raise ValueError("Angular geometry requires a finite 16-column center and positive bandwidth")
        return cls(center.copy(), h, copy.deepcopy(saved["audit"]))


def fit_angular_geometry(source_fit_features, source_groups):
    """Freeze response-key geometry using source FIT inputs only, no targets."""
    source = _features(source_fit_features, "source FIT features")
    groups = BL._labels(source_groups, "source FIT groups", len(source))
    center = BL.group_weights(groups) @ source[:, RNA_SLICE]
    unit = _unit_rows(source[:, RNA_SLICE] - center)
    angular_distance = _squared_distance(unit, unit) / 2.
    positive = angular_distance[(~np.eye(len(source), dtype=bool)) & (angular_distance > _EPS)]
    h = float(np.median(positive)) if len(positive) else 1.
    return AngularGeometry(center.copy(), h, dict(
        fit_rows=len(source), fit_groups=np.unique(groups).tolist(),
        block="frozen atlas RNA response-key coordinates 16:32",
        center="source FIT chemical-equal mean",
        bandwidth="median of positive off-diagonal source FIT half squared unit-or-zero distance; 1-cos for nonzero pairs",
        positive_pair_count=int(len(positive)), bandwidth_fallback=not bool(len(positive)),
        zero_centered_rows=int(np.sum(np.linalg.norm(unit, axis=1) == 0)),
        zero_policy="centered norm <= 1e-12 maps to zero; PSD squared-distance extension",
        reference_inputs_used=False, reference_targets_used=False, query_inputs_used=False,
        source_targets_used=False))


def _angular_features(features, geometry, angular_geometry):
    out = _features(features).copy()
    out[:, RNA_SLICE] = (_unit_rows(out[:, RNA_SLICE] - angular_geometry.response_center)
                        * np.sqrt(geometry["distance_reference"] / (2. * angular_geometry.bandwidth)))
    return out


def kernel_matrix(train, query, geometry, angular_geometry=None):
    """Query-by-train frozen AND kernel, optionally replacing only RNA RBF."""
    train, query = _features(train, "train features"), _features(query, "query features")
    geo = _geometry(geometry)
    if angular_geometry is not None:
        angular_geometry = AngularGeometry.from_dict(angular_geometry.to_dict())
        train = _angular_features(train, geo, angular_geometry)
        query = _angular_features(query, geo, angular_geometry)
    return _rbf(train, query, "rbf", BANDWIDTH, geo)


def _condition_means(residual, codes, groups):
    means, centered = {}, np.empty_like(residual)
    for code in np.unique(codes):
        mask = codes == code
        mean = BL.group_weights(groups[mask]) @ residual[mask]
        means[str(code)] = mean
        centered[mask] = residual[mask] - mean
    return means, centered


@dataclass
class MemoryModel:
    variant: str
    geometry: dict
    angular_geometry: AngularGeometry | None
    v15_model: V15MemoryModel | None
    shift_features: np.ndarray | None
    shift_coefficients: np.ndarray | None
    source_condition_means: dict
    reference_condition_means: dict
    audit: dict

    @property
    def dual_coefficients(self):
        return self.v15_model.dual_coefficients if self.v15_model is not None else self.shift_coefficients

    @property
    def train_features(self):
        return self.v15_model.train_features if self.v15_model is not None else self.shift_features

    def predict(self, query_features, query_codes=None):
        query = _features(query_features, "query features")
        if self.v15_model is not None:
            if self.angular_geometry is not None:
                query = _angular_features(query, self.geometry, self.angular_geometry)
            return self.v15_model.predict(query, np.ones(len(query)))
        codes = BL._labels(query_codes, "query condition codes", len(query))
        means = {**self.source_condition_means, **self.reference_condition_means}
        if set(codes) - set(means):
            raise ValueError("Query condition has neither a reference mean nor a source fallback mean")
        shift = np.stack([means[code] for code in codes])
        return shift + kernel_matrix(self.shift_features, query, self.geometry) @ self.shift_coefficients

    def to_dict(self):
        return dict(format="v16_memory", variant=self.variant, geometry=copy.deepcopy(self.geometry),
            angular_geometry=None if self.angular_geometry is None else self.angular_geometry.to_dict(),
            v15_model=None if self.v15_model is None else self.v15_model.to_dict(),
            shift_features=None if self.shift_features is None else self.shift_features.tolist(),
            shift_coefficients=None if self.shift_coefficients is None else self.shift_coefficients.tolist(),
            source_condition_means={k: v.tolist() for k, v in self.source_condition_means.items()},
            reference_condition_means={k: v.tolist() for k, v in self.reference_condition_means.items()},
            audit=copy.deepcopy(self.audit))

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v16_memory" or saved.get("variant") not in VARIANTS:
            raise ValueError("Unknown V16 memory format or variant")
        variant, geo = saved["variant"], _geometry(saved["geometry"])
        angular = (None if saved["angular_geometry"] is None else
                   AngularGeometry.from_dict(saved["angular_geometry"]))
        if (angular is not None) != (variant == "angular"):
            raise ValueError("Only the angular variant has angular geometry")
        old = None if saved["v15_model"] is None else V15MemoryModel.from_dict(saved["v15_model"])
        features = alpha = None
        source_means = {str(k): np.asarray(v, float) for k, v in saved["source_condition_means"].items()}
        reference_means = {str(k): np.asarray(v, float) for k, v in saved["reference_condition_means"].items()}
        if old is not None:
            if (old.kind != "and" or old.bandwidth != BANDWIDTH
                    or old.audit["lambda_value"] != LAMBDA or old.geometry != geo):
                raise ValueError("V16 requires frozen V15 AND geometry, bandwidth 1 and lambda 100")
        else:
            if variant != "shift":
                raise ValueError("Main/angular memory requires its V15 model")
            features = _features(saved["shift_features"], "saved shift features").copy()
            alpha = BL._matrix(saved["shift_coefficients"], "saved shift coefficients", len(features)).copy()
            if not source_means or not reference_means:
                raise ValueError("Shift memory requires source and reference condition means")
            if any(v.shape != (alpha.shape[1],) or not np.isfinite(v).all()
                   for v in list(source_means.values()) + list(reference_means.values())):
                raise ValueError("Saved condition means differ from the full output dimension")
        return cls(variant, geo, angular, old, features, alpha, source_means, reference_means,
                   copy.deepcopy(saved["audit"]))


def fit_memory(source_features, source_residual, source_groups, *, geometry,
               reference_features=None, reference_residual=None, reference_groups=None,
               variant="main", source_codes=None, reference_codes=None, angular_geometry=None):
    """Fit one declared residual arm, with fixed lambda=100 and bandwidth=1.

    Source and reference residuals must be relative to the same frozen G.
    Source rows precede reference rows exactly as in V15. References may be
    omitted or supplied as aligned empty arrays. Conditions are needed only
    for the shift arm with nonempty references. No query or query target is
    accepted by this fitting interface.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown V16 memory variant: {variant!r}")
    source = _features(source_features, "source features")
    residual = BL._matrix(source_residual, "source residual", len(source))
    groups = BL._labels(source_groups, "source groups", len(source))
    geo = _geometry(geometry)
    supplied = [v is not None for v in (reference_features, reference_residual, reference_groups)]
    if any(supplied) and not all(supplied):
        raise ValueError("Reference features, residuals and groups must be supplied together")
    refs, rr, rg = None, None, None
    if all(supplied):
        refs = np.asarray(reference_features, float)
        if len(refs) == 0:
            if (refs.shape != (0, source.shape[1])
                    or np.asarray(reference_residual).shape != (0, residual.shape[1])
                    or np.asarray(reference_groups).shape != (0,)):
                raise ValueError("Empty reference arrays have incompatible dimensions")
            refs = None
        else:
            refs = _features(refs, "reference features")
            rr = BL._matrix(reference_residual, "reference residual", len(refs))
            rg = BL._labels(reference_groups, "reference groups", len(refs))
            if rr.shape[1] != residual.shape[1]:
                raise ValueError("Source/reference residual output dimensions differ")
    features = source if refs is None else np.concatenate((source, refs))
    targets = residual if refs is None else np.concatenate((residual, rr))
    combined_groups = groups if refs is None else np.concatenate((groups, rg))
    if angular_geometry is not None and variant != "angular":
        raise ValueError("Only the angular variant accepts angular geometry")
    angular = None
    if variant == "angular":
        angular = (fit_angular_geometry(source, groups) if angular_geometry is None else
                   AngularGeometry.from_dict(angular_geometry.to_dict()))
        features = _angular_features(features, geo, angular)
    audit = dict(variant=variant, lambda_value=LAMBDA, bandwidth=BANDWIDTH,
        source_rows=len(source), reference_rows=0 if refs is None else len(refs),
        geometry_scope="caller-supplied frozen source FIT geometry",
        sample_weights="n * group_weights over combined source and references, exact V15 convention",
        query_inputs_used=False, query_targets_used=False, main_G_frozen=True,
        source_only_fallback=bool(variant == "shift" and refs is None))
    if variant != "shift" or refs is None:
        model = _fit_v15(features, targets, combined_groups, np.ones(len(features)),
                         geometry=geo, kind="and", lambda_value=LAMBDA, bandwidth=BANDWIDTH)
        audit["solver"] = copy.deepcopy(model.audit)
        return MemoryModel(variant, geo, angular, model, None, None, {}, {}, audit)
    sc = BL._labels(source_codes, "source condition codes", len(source))
    rc = BL._labels(reference_codes, "reference condition codes", len(refs))
    source_means, source_centered = _condition_means(residual, sc, groups)
    reference_means, reference_centered = _condition_means(rr, rc, rg)
    centered = np.concatenate((source_centered, reference_centered))
    weights = len(features) * BL.group_weights(combined_groups)
    sw = np.sqrt(weights)
    gram = kernel_matrix(features, features, geo)
    system = sw[:, None] * gram * sw[None, :]
    system = (system + system.T) * .5
    system.flat[::len(system) + 1] += LAMBDA
    rhs = sw[:, None] * centered
    beta = cho_solve(cho_factor(system, lower=True, check_finite=True), rhs, check_finite=True)
    alpha = np.ascontiguousarray(sw[:, None] * beta)
    audit.update(row_weights=weights.tolist(), row_weight_sum=float(weights.sum()),
        source_centering="source-domain chemical-equal mean separately for each source condition",
        target_shift="target-reference chemical-equal mean separately for each reference condition",
        query_shift_fallback_conditions=sorted(set(source_means) - set(reference_means)),
        global_intercept_fitted=False, source_reference_means_pooled=False,
        relative_normal_equation_error=float(np.linalg.norm(system @ beta - rhs) / max(1., np.linalg.norm(rhs))))
    return MemoryModel(variant, geo, None, None, features.copy(), alpha, source_means,
                       reference_means, audit)
