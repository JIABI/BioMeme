"""V14 inserted-reference residual memory and a pooled-layout sensitivity.

Memory uses the supplied frozen coordinates/geometry and supplied nonnegative
activity phi. Its normalizer q is chemical-equal E[phi²] over ALL supplied
memory entries (source FIT and inserted references), not over query inputs.
The floor median itself is not estimated here: it belongs to the frozen source
ActivityScale. The phi-proportional intercept is the V12 fixed, unpenalized
pre-fit projection, NOT a jointly optimized intercept of the kernel objective.

LayoutMultiwellCalibration is a distinct full-refit sensitivity. It learns
condition-specific slopes shared across layouts, after subtracting per-layout,
per-condition input/output means. It cannot predict an unseen layout, because
its target means require authorized reference outcomes for that layout.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from bioffn_v7 import baselines as BL
from bioffn_v8.memory import kernel_geometry
from bioffn_v10.memory import _geometry
from bioffn_v11.multiwell import MultiwellCalibration, ROUTES, _stream_arrays
from bioffn_v12.activity import ActivityMemoryModel, _phi, fit_activity_memory
from bioffn_v13.kernel import kernel_matrix


@dataclass
class MemoryModel(ActivityMemoryModel):
    kind: str = "and"

    def predict(self, features, phi):
        if self.kind == "and":
            # Preserve the predecessor's operation order, including its output
            # reconstruction and treatment of an all-zero activity population.
            return super().predict(features, phi)
        x = BL._matrix(features, "query features")
        phi = _phi(phi, len(x))
        h = phi/np.sqrt(self.q) if self.q > 0 else np.zeros_like(phi)
        base = kernel_matrix(self.train_features, x, self.kind, self.geometry, self.bandwidth)
        return (phi[:, None]*self.intercept_coefficients
                + h[:, None]*(base @ (self.train_h[:, None]*self.dual_coefficients)))

    def to_dict(self):
        result = super().to_dict()
        result.update(format="v14_activity_view_memory", kind=self.kind)
        return result

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v14_activity_view_memory":
            raise ValueError("Unknown V14 memory format")
        kind = saved.get("kind")
        if kind not in ("and", "morphology", "mechanism", "or"):
            raise ValueError("Unknown V14 kernel view")
        old = dict(saved, format="v12_activity_memory")
        base = ActivityMemoryModel.from_dict(old)
        if base.factor != "normalized" or base.intercept != "activity" or base.weight != "rbf":
            raise ValueError("V14 memory requires normalized activity and its fixed activity intercept")
        if base.train_features.shape[1] != 39:
            raise ValueError("V14 memory requires the frozen 39-coordinate schema")
        return cls(**vars(base), kind=kind)


def fit_memory(features, residual, groups, phi, geometry=None, kind="and", lambda_value=100.,
               bandwidth=1.):
    """Fit and return a model with ``predict(query_features, query_phi)``.

With g summing to one, q=sum g*phi², h=phi/sqrt(q), W=n*diag(g),
b=(g*phi)'R/q and T=R-phi*b, solve
  (sqrt(W) D_h K D_h sqrt(W)+lambda I)c=sqrt(W)T,
then alpha=sqrt(W)c. Query output is phi_q*b+h_q*K_q*D_h*alpha.
Constant phi=1 recovers chemical-equal centered V11 memory. No memory source
is implicitly reweighted: all supplied chemical groups have equal total mass.
    """
    x = BL._matrix(features, "memory features")
    if x.shape[1] != 39:
        raise ValueError("V14 memory requires the frozen 39-coordinate schema")
    if kind not in ("and", "morphology", "mechanism", "or"):
        raise ValueError("Unknown V14 kernel view")
    if kind == "and":
        base = fit_activity_memory(x, residual, groups, phi, factor="normalized", intercept="activity",
            lambda_value=lambda_value, geometry=geometry, weight="rbf", bandwidth=bandwidth)
    else:
        residual = BL._matrix(residual, "residual", len(x))
        groups = BL._labels(groups, "groups", len(x))
        phi = _phi(phi, len(x))
        if not np.isfinite(lambda_value) or lambda_value <= 0:
            raise ValueError("Ridge lambda must be finite and positive")
        gw = BL.group_weights(groups)
        with np.errstate(over="ignore", invalid="ignore"):
            q = float(gw @ (phi*phi))
        if not np.isfinite(q):
            raise ValueError("Activity second moment is outside the finite numeric range")
        h = phi/np.sqrt(q) if q > 0 else np.zeros_like(phi)
        b = (gw*phi) @ residual/q if q > 0 else np.zeros(residual.shape[1])
        centered = residual-phi[:, None]*b
        geo = _geometry(kernel_geometry(x) if geometry is None else geometry)
        k = kernel_matrix(x, x, kind, geo, bandwidth)*h[:, None]*h[None, :]
        w = len(x)*gw
        sw = np.sqrt(w)
        system = sw[:, None]*k*sw[None, :]
        system = (system+system.T)*.5
        system.flat[::len(system)+1] += lambda_value
        rhs = sw[:, None]*centered
        beta = cho_solve(cho_factor(system, lower=True, check_finite=True), rhs, check_finite=True)
        alpha = np.ascontiguousarray(sw[:, None]*beta)
        audit = dict(method="activity-modulated chemical-equal full-output view KRR", rows=len(x),
            groups=len(np.unique(groups)), fit_groups=np.unique(groups).tolist(), full_output_dim=residual.shape[1],
            factor="normalized", intercept="activity", lambda_value=float(lambda_value), weight="rbf",
            bandwidth=float(bandwidth), geometry=geo, activity_second_moment=q,
            row_weight_sum=float(w.sum()), row_weights=w.tolist(),
            activity_factor_second_moment=float(gw @ (h*h)), zero_activity_rows=int(np.sum(phi == 0)),
            kernel_formula="diag(h) K_view diag(h)", h_formula="phi/sqrt(sum memory_group_weight*phi^2)",
            intercept_formula="phi*((group_weight*phi)@R)/q",
            intercept_estimation="pre-fit weighted projection; fixed during kernel solve",
            input_lambda_unchanged=True, regularization_normalization="chemical-weighted E[h²]=1 when q>0",
            raw_kernel_equivalent_lambda=float(lambda_value*q),
            unit_residual_equivalent_weights=(w*phi*phi/q).tolist() if q > 0 else None,
            raw_coordinate_solve=True, epsilon_division=False, zero_second_moment=q == 0,
            relative_normal_equation_error=float(np.linalg.norm(system@beta-rhs)/max(1., np.linalg.norm(rhs))))
        base = ActivityMemoryModel(x.copy(), alpha, h, b.copy(), "normalized", "activity", q,
                                   geo, "rbf", float(bandwidth), audit)
    audit = copy.deepcopy(base.audit)
    audit.update(kind=kind, normalization_scope="all supplied memory entries: source plus inserted references",
        phi_scope="caller-supplied, from frozen source ActivityScale; no floor median fitted here",
        geometry_scope="caller-supplied frozen full-feature geometry" if geometry is not None else "supplied memory features",
        source_reference_mass="chemical-equal over the combined memory; n*g sum convention",
        query_inputs_used_to_fit=False, fixed_intercept_not_jointly_optimized=True)
    base.audit = audit
    return MemoryModel(**vars(base), kind=kind)


class LayoutMultiwellCalibration:
    """Shared condition slopes, with layout-specific affine intercepts/centers."""

    def fit(self, streams, active, y, codes, groups, layouts):
        streams, active, codes = _stream_arrays(streams, active, codes)
        y = BL._matrix(y, "target", len(streams))
        groups = BL._labels(groups, "groups", len(streams))
        layouts = BL._labels(layouts, "layouts", len(streams))
        if y.shape[1] != streams.shape[2]:
            raise ValueError("Source and output feature dimensions differ")
        self.n_features = y.shape[1]
        self.source_mean, self.target_mean, self.coefficients, self.active_patterns = {}, {}, {}, {}
        # This path preserves V11 numerical identity, not merely its objective.
        unique_layouts = np.unique(layouts)
        if len(unique_layouts) == 1:
            ordinary = MultiwellCalibration().fit(streams, active, y, codes, groups)
            layout = str(unique_layouts[0])
            self.source_mean = {c: {layout: m.copy()} for c, m in ordinary.source_mean.items()}
            self.target_mean = {c: {layout: m.copy()} for c, m in ordinary.target_mean.items()}
            self.coefficients = {c: m.copy() for c, m in ordinary.coefficients.items()}
            self.active_patterns = {c: m.copy() for c, m in ordinary.active_patterns.items()}
            self.audit = copy.deepcopy(ordinary.audit)
            self.audit.update(method="layout-centered shared-slope multiwell OLS", layouts=[layout],
                source_output_means="layout and condition specific", slopes="condition specific; shared across layouts",
                single_layout_exact_V11=True, query_means_used=False)
            return self
        details = {}
        for code in np.unique(codes):
            mask = codes == code
            pattern = active[mask][0].copy()
            routes = np.flatnonzero(pattern)
            x, target, lay = streams[mask][:, routes, :], y[mask], layouts[mask]
            w = BL.group_weights(groups[mask])
            xc, yc = np.empty_like(x), np.empty_like(target)
            source_means, target_means, layout_details = {}, {}, {}
            for layout in np.unique(lay):
                inside = lay == layout
                within_w = w[inside]/w[inside].sum()
                mean = np.einsum("n,nkd->kd", within_w, x[inside])
                target_mean = within_w @ target[inside]
                if len(routes) == 1:
                    mean[0] = within_w @ x[inside, 0]
                xc[inside], yc[inside] = x[inside]-mean, target[inside]-target_mean
                full_mean = np.zeros((4, self.n_features))
                full_mean[routes] = mean
                source_means[str(layout)], target_means[str(layout)] = full_mean, target_mean
                layout_details[str(layout)] = dict(rows=int(inside.sum()),
                    groups=len(np.unique(groups[mask][inside])), fit_groups=np.unique(groups[mask][inside]).tolist(),
                    condition_weight_mass=float(w[inside].sum()))
            sw = len(w)*w
            if len(routes) == 1:
                xx = sw @ (xc[:, 0]*xc[:, 0])
                xy = sw @ (xc[:, 0]*yc)
                denominator = float(xx.sum())
                fitted = np.array([float(xy.sum()/denominator) if denominator > 1e-12 else 0.])
                rank, singular = int(denominator > 1e-12), np.array([np.sqrt(max(denominator, 0.))])
            else:
                root_w = np.sqrt(sw)
                design = (root_w[:, None, None]*xc).transpose(0, 2, 1).reshape(-1, len(routes))
                response = (root_w[:, None]*yc).reshape(-1)
                fitted, _, rank, singular = np.linalg.lstsq(design, response, rcond=None)
            coefficients = np.zeros(4)
            coefficients[routes] = fitted
            self.source_mean[str(code)], self.target_mean[str(code)] = source_means, target_means
            self.coefficients[str(code)], self.active_patterns[str(code)] = coefficients, pattern
            details[str(code)] = dict(rows=int(mask.sum()), groups=len(np.unique(groups[mask])),
                fit_groups=np.unique(groups[mask]).tolist(), layouts=layout_details,
                active_routes=[ROUTES[j] for j in routes], coefficients=coefficients.tolist(),
                design_rank=int(rank), design_singular_values=singular.tolist(),
                weighted_sum_normalization=float(sw.sum()))
        self.audit = dict(method="layout-centered shared-slope multiwell OLS", fit_rows=len(streams),
            fit_groups=np.unique(groups).tolist(), layouts=unique_layouts.tolist(), output_dim=self.n_features,
            formula="mu_y(layout,condition)+sum_j b(condition,j)*(x_j-mu_x(layout,condition,j))",
            source_output_means="layout and condition specific", slopes="condition specific; shared across layouts",
            means_weighting="within-layout restriction of condition chemical-equal weights, renormalized",
            slope_weighting="chemical-equal within condition over all layouts; no equal-layout weighting",
            coefficients_unconstrained=True, per_feature_coefficients=False, conditions=details,
            single_layout_exact_V11=False, query_means_used=False)
        return self

    def predict(self, streams, active, codes, layouts):
        if not hasattr(self, "n_features"):
            raise ValueError("Calibration has not been fitted")
        streams, active, codes = _stream_arrays(streams, active, codes)
        layouts = BL._labels(layouts, "query layouts", len(streams))
        if streams.shape[2] != self.n_features or set(codes)-set(self.source_mean):
            raise ValueError("Prediction has an unseen condition or different feature dimension")
        result = np.empty((len(streams), self.n_features))
        for code in np.unique(codes):
            if not np.array_equal(active[codes == code][0], self.active_patterns[code]):
                raise ValueError("Prediction active routes differ from fitted condition")
            routes = np.flatnonzero(self.active_patterns[code])
            for layout in np.unique(layouts[codes == code]):
                if layout not in self.source_mean[code]:
                    raise ValueError("Prediction layout lacks fitted target/source means for this condition")
                mask = (codes == code) & (layouts == layout)
                mean, target_mean = self.source_mean[code][layout], self.target_mean[code][layout]
                if len(routes) == 1:
                    j = routes[0]
                    result[mask] = target_mean+self.coefficients[code][j]*(streams[mask, j]-mean[j])
                else:
                    centered = streams[mask][:, routes, :]-mean[routes]
                    result[mask] = target_mean+np.einsum("j,njd->nd", self.coefficients[code][routes], centered)
        return result

    def to_dict(self):
        if not hasattr(self, "n_features"):
            raise ValueError("Calibration has not been fitted")
        return dict(format="v14_layout_multiwell", n_features=self.n_features,
            source_mean={c: {l: x.tolist() for l,x in vals.items()} for c,vals in self.source_mean.items()},
            target_mean={c: {l: x.tolist() for l,x in vals.items()} for c,vals in self.target_mean.items()},
            coefficients={c: x.tolist() for c,x in self.coefficients.items()},
            active_patterns={c: x.tolist() for c,x in self.active_patterns.items()}, audit=copy.deepcopy(self.audit))

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v14_layout_multiwell":
            raise ValueError("Unknown layout-calibration format")
        result = cls()
        result.n_features = int(saved["n_features"])
        if result.n_features < 1:
            raise ValueError("Invalid saved feature dimension")
        for name in ("source_mean", "target_mean"):
            setattr(result, name, {str(c): {str(l): np.asarray(x, float) for l,x in vals.items()}
                                  for c,vals in saved[name].items()})
        result.coefficients = {str(c): np.asarray(x, float) for c,x in saved["coefficients"].items()}
        result.active_patterns = {str(c): np.asarray(x, bool) for c,x in saved["active_patterns"].items()}
        keys = set(result.source_mean)
        if not keys or any(set(getattr(result, name)) != keys for name in ("target_mean", "coefficients", "active_patterns")):
            raise ValueError("Saved condition inventories differ")
        for code in keys:
            pattern, coefficients = result.active_patterns[code], result.coefficients[code]
            if (pattern.shape != (4,) or not pattern.any() or coefficients.shape != (4,)
                    or not np.isfinite(coefficients).all() or np.any(coefficients[~pattern] != 0)
                    or not result.source_mean[code]
                    or set(result.source_mean[code]) != set(result.target_mean[code])):
                raise ValueError("Invalid saved condition parameters")
            for layout in result.source_mean[code]:
                x, y = result.source_mean[code][layout], result.target_mean[code][layout]
                if (x.shape != (4, result.n_features) or y.shape != (result.n_features,)
                        or not np.isfinite(x).all() or not np.isfinite(y).all()):
                    raise ValueError("Invalid saved layout means")
        result.audit = copy.deepcopy(saved["audit"])
        return result


def crossfit_layout_multiwell(streams, active, y, codes, groups, layouts, n_folds=8, seed=20261008+200009):
    """Whole-chemical OOF, balanced within each observed layout.

Every chemical must belong to one layout. The shared fold count is capped by
the smallest layout's chemical count. At least two chemicals per layout are
needed; every held-out condition/layout must retain training references.
    """
    streams, active, codes = _stream_arrays(streams, active, codes)
    y = BL._matrix(y, "target", len(streams))
    groups = BL._labels(groups, "groups", len(streams))
    layouts = BL._labels(layouts, "layouts", len(streams))
    if isinstance(n_folds, bool) or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    if any(len(np.unique(layouts[groups == g])) != 1 for g in np.unique(groups)):
        raise ValueError("Stratified layout cross-fitting requires each chemical to belong to one layout")
    unique_layouts = np.unique(layouts)
    minimum = min(len(np.unique(groups[layouts == lay])) for lay in unique_layouts)
    if minimum < 2:
        raise ValueError("Every layout needs at least two chemical groups for honest layout-mean estimation")
    count = min(int(n_folds), minimum)
    fid = np.empty(len(streams), dtype=int)
    for index, layout in enumerate(unique_layouts):
        mask = layouts == layout
        fid[mask] = BL.make_folds(groups[mask], count, int(seed)+104729*index)
    predicted, audits = np.empty_like(y), []
    for fold in np.unique(fid):
        train, valid = fid != fold, fid == fold
        required = set(zip(layouts[valid], codes[valid]))
        available = set(zip(layouts[train], codes[train]))
        if required-available:
            raise ValueError("An OOF layout/condition has no remaining reference chemical")
        model = LayoutMultiwellCalibration().fit(streams[train], active[train], y[train],
                                                 codes[train], groups[train], layouts[train])
        predicted[valid] = model.predict(streams[valid], active[valid], codes[valid], layouts[valid])
        audits.append(dict(fold=int(fold), train_indices=np.flatnonzero(train).tolist(),
            validation_indices=np.flatnonzero(valid).tolist(), train_groups=np.unique(groups[train]).tolist(),
            validation_groups=np.unique(groups[valid]).tolist(), train_layouts=np.unique(layouts[train]).tolist(),
            validation_layouts=np.unique(layouts[valid]).tolist(), baseline=model.audit))
    return predicted, dict(folds=audits, actual_folds=len(audits), seed=int(seed),
        fold_assignment=fid.tolist(), stratification="chemical groups within layout",
        all_rows_predicted_once=True, group_leakage=False,
        means_and_coefficients_fit_on_training_fold_only=True)
