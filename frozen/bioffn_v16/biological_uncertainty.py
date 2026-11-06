"""Frozen-model uncertainty scores and oracle measurement-replacement analysis.

No fitting, calibration, model selection, or prospective measurement simulation is
performed here. The KRR score omits fixed-intercept and coordinate uncertainty.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr


def _vector(x, n, label):
    out = np.asarray(x, float)
    if out.shape != (n,) or not np.isfinite(out).all():
        raise ValueError(f"{label} must be an aligned finite vector")
    return out


def posterior_variance(train_kernel, query_kernel, query_diagonal, *,
                       row_weights, train_h, query_h, ridge=100.):
    """Latent weighted-KRR variance using the saved diagonal factors exactly.

    A = sqrt(W) diag(h) K diag(h) sqrt(W) + ridge I.
    Vq = hq^2 kqq - (hq kq diag(h) sqrt(W)) A^-1 (...)^T.
    Weights are normalized to mean one by the original fitter, not here.
    """
    k = np.asarray(train_kernel, float)
    cross = np.asarray(query_kernel, float)
    if (k.ndim != 2 or k.shape[0] != k.shape[1] or cross.ndim != 2
            or cross.shape[1] != len(k) or not np.isfinite(k).all()
            or not np.isfinite(cross).all()):
        raise ValueError("Invalid Gram/cross-kernel matrices")
    n, nq = len(k), len(cross)
    weights = _vector(row_weights, n, "weights")
    h = _vector(train_h, n, "train h")
    hq = _vector(query_h, nq, "query h")
    diagonal = _vector(query_diagonal, nq, "query diagonal")
    if (np.any(weights <= 0) or np.any(diagonal < 0)
            or not np.isclose(weights.mean(), 1., rtol=1e-12, atol=1e-12)
            or not np.isfinite(ridge) or ridge <= 0
            or not np.allclose(k, k.T, rtol=1e-10, atol=1e-10)):
        raise ValueError("Invalid KRR weights, diagonal, ridge, or symmetry")
    factor = np.sqrt(weights)*h
    system = factor[:, None]*k*factor[None, :]
    system = .5*(system+system.T)
    system.flat[::n+1] += ridge
    weighted_cross = hq[:, None]*cross*factor[None, :]
    solution = cho_solve(cho_factor(system, lower=True), weighted_cross.T)
    prior = hq**2*diagonal
    reduction = np.einsum("ij,ji->i", weighted_cross, solution)
    value = prior-reduction
    tolerance = 1e-9*max(1., float(np.max(prior)))
    if value.min(initial=0.) < -tolerance:
        raise ValueError("Materially negative posterior variance")
    return np.maximum(value, 0.), dict(ridge=float(ridge),
        weight_min=float(weights.min()), weight_max=float(weights.max()),
        weight_mean=float(weights.mean()), h_min=float(h.min()), h_max=float(h.max()),
        min_before_numerical_clipping=float(value.min()), prior_min=float(prior.min()),
        prior_max=float(prior.max()), max_reduction=float(reduction.max()))


def chemical_mean(values, groups):
    values = np.asarray(values, float); groups = np.asarray(groups, str)
    if values.shape != groups.shape or not np.isfinite(values).all():
        raise ValueError("Values/groups must be finite aligned vectors")
    labels = np.unique(groups)
    means = np.array([values[groups == g].mean() for g in labels])
    return labels, means


def risk_coverage(score, row_error, groups, ids, *, coverages=(.5, .7, .9, 1.)):
    """Retain the lowest scores, with stable row-ID ties; no outcome selection."""
    ids = np.asarray(ids, str); groups = np.asarray(groups, str)
    n = len(ids); score = _vector(score, n, "uncertainty")
    error = _vector(row_error, n, "row error")
    if len(set(ids)) != n or groups.shape != (n,):
        raise ValueError("IDs must be unique and groups aligned")
    order = np.lexsort((ids, score))
    curves = []
    for fraction in coverages:
        if not 0 < fraction <= 1: raise ValueError("Coverage must be in (0,1]")
        take = order[:int(np.ceil(fraction*n))]
        chemicals, means = chemical_mean(error[take], groups[take])
        curves.append(dict(requested_coverage=float(fraction), rows=len(take),
            actual_coverage=len(take)/n, chemical_count=len(chemicals),
            mse=float(error[take].mean()), chemical_equal_mse=float(means.mean())))
    rho = (float(spearmanr(score, error).statistic)
           if np.ptp(score) > 0 and np.ptp(error) > 0 else None)
    return dict(spearman_uncertainty_vs_row_mse=rho, curves=curves,
        uncertainty_range=[float(score.min()), float(score.max())],
        tie_rule="Ascending uncertainty then stable query row ID", inferential_status="descriptive")


def seed_variance(predictions):
    """Average-feature sample variance of predictions across seeds (ddof=1)."""
    pred = np.asarray(predictions, float)
    if pred.ndim != 3 or len(pred) < 2 or not np.isfinite(pred).all():
        raise ValueError("At least two finite seed×row×feature predictions required")
    return np.var(pred, axis=0, ddof=1).mean(axis=1)


def _chemical_inventory(groups, layouts, ids):
    groups = np.asarray(groups, str); layouts = np.asarray(layouts, str); ids = np.asarray(ids, str)
    if groups.shape != layouts.shape or groups.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError("Aligned inventory and unique IDs required")
    chemicals = np.unique(groups)
    masks, chemical_layouts, tie_ids = [], [], []
    for group in chemicals:
        take = np.flatnonzero(groups == group)
        if len(take) != 4 or len(set(layouts[take])) != 1:
            raise ValueError("Each query chemical must have four rows in exactly one layout")
        masks.append(take); chemical_layouts.append(layouts[take[0]])
        tie_ids.append(min(ids[take]))
    return chemicals, masks, np.array(chemical_layouts), np.array(tie_ids)


def reference_selections(scores, groups, layouts, ids, *, budgets=(.1, .2, .3),
                         random_repeats=100, random_seed=20261013):
    """Nested whole-chemical measurement lists, independently budgeted by layout."""
    chemicals, masks, clayouts, tie_ids = _chemical_inventory(groups, layouts, ids)
    names = sorted(set(clayouts)); orders = {}
    for name, values in scores.items():
        row = _vector(values, len(ids), name)
        chemical_scores = np.array([row[t].mean() for t in masks])
        orders[name] = [{layout: np.flatnonzero(clayouts == layout)[
            np.lexsort((tie_ids[clayouts == layout], -chemical_scores[clayouts == layout]))]
            for layout in names}]
    rng = np.random.default_rng(random_seed)
    orders['random'] = [{layout: rng.permutation(np.flatnonzero(clayouts == layout))
                        for layout in names} for _ in range(random_repeats)]
    result = {}
    for name, rankings in orders.items():
        result[name] = {}
        for budget in (0.,)+tuple(budgets):
            if not 0 <= budget <= 1: raise ValueError("Budget must be in [0,1]")
            repeated = []
            for ranking in rankings:
                selected = np.concatenate([ranking[layout][:
                    int(np.ceil(budget*len(ranking[layout])))] for layout in names])
                repeated.append(chemicals[selected])
            result[name][str(budget)] = repeated
    return result


def replacement_curve(row_error, predicted_active, target_active, selections,
                      groups, layouts, codes, contexts, ids):
    """Replace selected predictions by the existing measured target, no retraining.

    Error and activity recall are scored for every seed separately, then averaged.
    Thus the output does not introduce a mean-prediction ensemble model.
    """
    ids = np.asarray(ids, str); groups = np.asarray(groups, str)
    layouts = np.asarray(layouts, str); codes = np.asarray(codes, str)
    contexts = np.asarray(contexts, str); n = len(ids)
    error = np.asarray(row_error, float); active = np.asarray(predicted_active, bool)
    target = np.asarray(target_active, bool)
    if error.ndim == 1: error = error[None, :]
    if active.ndim == 1: active = active[None, :]
    if (error.ndim != 2 or error.shape != active.shape or error.shape[1] != n
            or target.shape != (n,) or not np.isfinite(error).all()
            or codes.shape != (n,) or contexts.shape != (n,)):
        raise ValueError("Seed scores and activity masks are not aligned")
    chemicals, masks, clayouts, _ = _chemical_inventory(groups, layouts, ids)
    true_chem = np.array([target[t].any() for t in masks])
    galleries = np.array([f"{c}/{t}" for c, t in zip(contexts, codes)])
    gallery_names = np.unique(galleries)
    output = {}
    for selector, budgets in selections.items():
        output[selector] = {}
        for budget, repeated in budgets.items():
            reps = []
            for selected in repeated:
                selected = np.asarray(selected, str)
                if len(set(selected)) != len(selected) or not set(selected) <= set(chemicals):
                    raise ValueError("Invalid selected chemical list")
                take = np.isin(groups, selected); chemical_take = np.isin(chemicals, selected)
                e = error.copy(); e[:, take] = 0.
                pred = active.copy(); pred[:, take] = target[take]
                chemical_error = np.array([e[:, t].mean(axis=1) for t in masks]).T
                pred_chemical = np.array([pred[:, t].any(axis=1) for t in masks]).T
                recall = (pred_chemical[:, true_chem].mean(axis=1)
                          if true_chem.any() else np.full(len(error), np.nan))
                by_gallery = {}
                for gallery in gallery_names:
                    positive = (galleries == gallery)&target
                    by_gallery[gallery] = (float(pred[:, positive].mean()) if positive.any() else None)
                by_layout = {}
                for layout in np.unique(clayouts):
                    cmask = clayouts == layout; positives = cmask & true_chem
                    by_layout[layout] = dict(chemicals=int(cmask.sum()),
                        selected=int(np.sum(cmask & chemical_take)),
                        chemical_equal_mse=float(chemical_error[:, cmask].mean()),
                        active_chemical_recall=float(pred_chemical[:, positives].mean()) if positives.any() else None)
                reps.append(dict(chemical_equal_mse=float(chemical_error.mean()),
                    row_mse=float(e.mean()), active_chemical_recall=float(recall.mean()) if true_chem.any() else None,
                    row_active_recall=float(pred[:, target].mean()) if target.any() else None,
                    gallery_row_active_recall=by_gallery, by_layout=by_layout,
                    selected_chemical_count=len(selected), selected_row_count=int(take.sum()),
                    selected_chemicals=selected.tolist()))
            fields = ('chemical_equal_mse', 'row_mse', 'active_chemical_recall', 'row_active_recall')
            summary = {}
            for field in fields:
                values = [r[field] for r in reps if r[field] is not None]
                summary[field] = (dict(mean=float(np.mean(values)),
                    min=float(np.min(values)), max=float(np.max(values)),
                    randomization_interval_95=np.quantile(values, [.025, .975]).tolist()) if values else None)
            output[selector][budget] = dict(summary=summary, repeats=reps,
                repeat_count=len(reps), active_chemical_count=int(true_chem.sum()),
                total_chemical_count=len(chemicals))
    return output
