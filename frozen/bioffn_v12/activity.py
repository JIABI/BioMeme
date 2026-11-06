"""FIT-only activity scales and activity-modulated full-output residual KRR.

Let g_i be chemical-equal row weights summing to one, q=sum g_i phi_i²,
and W=n diag(g). The normalized kernel is D_h K D_h, h=phi/sqrt(q).
The activity intercept is phi*b, b=sum g_i phi_i R_i/q. Intercept estimation
is a fixed preprocessing step, not joint unpenalized-intercept optimization.
The equivalent unit-residual fit uses weights n*g*phi²/q and the SAME ridge
lambda. All fitting remains in raw response coordinates, including phi=0.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from bioffn_v7 import baselines as BL
from bioffn_v8.memory import kernel_geometry, kernel_matrix
from bioffn_v10.memory import _geometry, _kernel_parameters


def _phi(value, n):
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Activity must be a finite nonnegative aligned vector")
    return result


@dataclass
class ActivityScale:
    means: dict
    median_activity: float
    n_features: int
    audit: dict

    @classmethod
    def fit(cls, x_same, codes, groups):
        x = BL._matrix(x_same, "same-time source morphology")
        codes = BL._labels(codes, "codes", len(x))
        groups = BL._labels(groups, "groups", len(x))
        means, raw, details = {}, np.empty(len(x)), {}
        for code in np.unique(codes):
            mask = codes == code
            weight = BL.group_weights(groups[mask])
            mean = weight @ x[mask]
            means[str(code)] = mean
            raw[mask] = np.sqrt(np.mean(np.square(x[mask]-mean), axis=1))
            details[str(code)] = dict(rows=int(mask.sum()), groups=len(np.unique(groups[mask])),
                fit_groups=np.unique(groups[mask]).tolist(), mean_rms=float(np.sqrt(np.mean(mean**2))))
        median = float(np.median(raw))
        audit = dict(method="same-time source morphology activity around FIT condition mean",
            fit_rows=len(x), fit_groups=np.unique(groups).tolist(), conditions=details,
            raw_formula="sqrt(mean_features((x-mu_condition_FIT)^2))",
            floor_formula="sqrt(raw^2+(global_FIT_median_raw/2)^2)",
            global_FIT_median_raw=median, floor_scale=median/2,
            median_weighting="ordinary median across FIT rows; condition means are chemical-equal",
            outcomes_used=False, query_moments_used=False)
        return cls(means, median, x.shape[1], audit)

    def transform(self, x_same, codes, mode="floor"):
        if mode not in ("linear", "floor"):
            raise ValueError("Activity transform must be linear or floor")
        x = BL._matrix(x_same, "query same-time source morphology")
        codes = BL._labels(codes, "query codes", len(x))
        if x.shape[1] != self.n_features or set(codes)-set(self.means):
            raise ValueError("Activity query dimensions/conditions differ from FIT")
        raw = np.empty(len(x))
        for code in np.unique(codes):
            mask = codes == code
            raw[mask] = np.sqrt(np.mean(np.square(x[mask]-self.means[code]), axis=1))
        phi = raw.copy() if mode == "linear" else np.hypot(raw, self.median_activity/2)
        return {"raw": raw, "phi": phi, "mode": mode}

    def to_dict(self):
        return dict(format="v12_activity_scale", means={k: v.tolist() for k,v in self.means.items()},
            median_activity=self.median_activity, n_features=self.n_features, audit=copy.deepcopy(self.audit))

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v12_activity_scale":
            raise ValueError("Unknown activity scale format")
        width, median = int(saved["n_features"]), float(saved["median_activity"])
        means = {str(k): np.asarray(v, float) for k,v in saved["means"].items()}
        if (width < 1 or not means or not np.isfinite(median) or median < 0
                or any(v.shape != (width,) or not np.isfinite(v).all() for v in means.values())):
            raise ValueError("Invalid saved activity scale")
        return cls(means, median, width, copy.deepcopy(saved["audit"]))


@dataclass
class ActivityMemoryModel:
    train_features: np.ndarray
    dual_coefficients: np.ndarray
    train_h: np.ndarray
    intercept_coefficients: np.ndarray
    factor: str
    intercept: str
    q: float
    geometry: dict
    weight: object
    bandwidth: float
    audit: dict

    def predict(self, qfeatures, qphi):
        features = BL._matrix(qfeatures, "query features")
        phi = _phi(qphi, len(features))
        h = _factor(phi, self.factor, self.q)
        base = kernel_matrix(self.train_features, features, self.weight, self.bandwidth, self.geometry)
        predicted = h[:, None]*(base @ (self.train_h[:, None]*self.dual_coefficients))
        if self.intercept == "activity":
            predicted += phi[:, None]*self.intercept_coefficients
        else:
            predicted += self.intercept_coefficients
        return predicted

    def to_dict(self):
        return dict(format="v12_activity_memory", train_features=self.train_features.tolist(),
            dual_coefficients=self.dual_coefficients.tolist(), train_h=self.train_h.tolist(),
            intercept_coefficients=self.intercept_coefficients.tolist(), factor=self.factor,
            intercept=self.intercept, q=self.q, geometry=copy.deepcopy(self.geometry),
            weight=self.weight, bandwidth=self.bandwidth, audit=copy.deepcopy(self.audit))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v12_activity_memory":
            raise ValueError("Unknown activity memory format")
        x = BL._matrix(saved["train_features"], "saved features").copy()
        alpha = BL._matrix(saved["dual_coefficients"], "saved dual coefficients", len(x)).copy()
        h = _phi(saved["train_h"], len(x)).copy()
        b = np.asarray(saved["intercept_coefficients"], float)
        q = float(saved["q"])
        _mode(saved["factor"], saved["intercept"])
        _kernel_parameters(saved["weight"], saved["bandwidth"])
        if b.shape != (alpha.shape[1],) or not np.isfinite(b).all() or not np.isfinite(q) or q < 0:
            raise ValueError("Invalid saved activity memory")
        return cls(x, alpha, h, b.copy(), saved["factor"], saved["intercept"], q,
            _geometry(saved["geometry"]), saved["weight"], float(saved["bandwidth"]), copy.deepcopy(saved["audit"]))


def _mode(factor, intercept):
    if factor not in ("normalized", "raw", "none"):
        raise ValueError("Kernel activity factor must be normalized, raw or none")
    if intercept not in ("activity", "constant"):
        raise ValueError("Intercept must be activity or constant")


def _factor(phi, mode, q):
    if mode == "none":
        return np.ones_like(phi)
    if mode == "raw":
        return phi.copy()
    return phi/np.sqrt(q) if q > 0 else np.zeros_like(phi)


def fit_activity_memory(features, residual, groups, phi, *, factor="normalized", intercept="activity",
                        lambda_value=100., geometry=None, weight="rbf", bandwidth=1.):
    """Full-output chemical-equal KRR with independently chosen factor/intercept."""
    x = BL._matrix(features, "features")
    residual = BL._matrix(residual, "residual", len(x))
    groups = BL._labels(groups, "groups", len(x))
    phi = _phi(phi, len(x))
    _mode(factor, intercept)
    _kernel_parameters(weight, bandwidth)
    if not np.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("Ridge lambda must be finite and positive")
    gw = BL.group_weights(groups)
    with np.errstate(over="ignore", invalid="ignore"):
        q = float(gw @ (phi*phi))
    if not np.isfinite(q):
        raise ValueError("Activity second moment is outside the finite numeric range")
    h = _factor(phi, factor, q)
    if intercept == "activity":
        b = (gw*phi) @ residual/q if q > 0 else np.zeros(residual.shape[1])
        centered = residual-phi[:, None]*b
    else:
        b = gw @ residual
        centered = residual-b
    geo = _geometry(kernel_geometry(x) if geometry is None else geometry)
    kernel = kernel_matrix(x, x, weight, bandwidth, geo)*h[:, None]*h[None, :]
    w = len(x)*gw
    sw = np.sqrt(w)
    system = sw[:, None]*kernel*sw[None, :]
    system = (system+system.T)*.5
    system.flat[::len(system)+1] += lambda_value
    rhs = sw[:, None]*centered
    alpha_weighted = cho_solve(cho_factor(system, lower=True, check_finite=True), rhs, check_finite=True)
    alpha = np.ascontiguousarray(sw[:, None]*alpha_weighted)
    audit = dict(method="activity-modulated chemical-equal full-output KRR", rows=len(x),
        groups=len(np.unique(groups)), fit_groups=np.unique(groups).tolist(), full_output_dim=residual.shape[1],
        factor=factor, intercept=intercept, lambda_value=float(lambda_value), weight=weight,
        bandwidth=float(bandwidth), geometry=geo, activity_second_moment=q,
        row_weight_sum=float(w.sum()), row_weights=w.tolist(),
        activity_factor_second_moment=float(gw @ (h*h)), zero_activity_rows=int(np.sum(phi == 0)),
        kernel_formula="diag(h) K_base diag(h)",
        h_formula={"normalized": "phi/sqrt(sum FIT_group_weight*phi^2)", "raw": "phi", "none": "1"}[factor],
        intercept_formula="phi*((group_weight*phi)@R)/q" if intercept == "activity" else "group_weight@R",
        intercept_estimation="pre-fit weighted projection; fixed during kernel solve",
        input_lambda_unchanged=True, regularization_normalization="normalized factor has chemical-weighted E[h^2]=1 when q>0",
        raw_kernel_equivalent_lambda=float(lambda_value*q) if factor == "normalized" else None,
        unit_residual_equivalent_weights=(w*phi*phi/q).tolist() if factor == "normalized" and q > 0 else None,
        raw_coordinate_solve=True, epsilon_division=False, zero_second_moment=q == 0,
        relative_normal_equation_error=float(np.linalg.norm(system@alpha_weighted-rhs)/max(1., np.linalg.norm(rhs))))
    return ActivityMemoryModel(x.copy(), alpha, h, b.copy(), factor, intercept, q, geo,
                               weight, float(bandwidth), audit)
