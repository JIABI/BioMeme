"""K4c reliability gating of the conserved term in the K4b transfer head.

Only beta's regressor is gated: X_c = delta * e / (e + c), with
e = mean(delta**2) per row.  The PCA basis and trajectory transformation
continue to use the original donor delta.  c=0 is the K4b candidate.
Selection uses each family's own saved OOF errors and fold-local beta fits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crosscell import _inventory, _labels, _matrix, validate_oof_coverage
from .transfer import Basis, LAMBDAS, RANKS, TransferModel, chemical_weights, scalar_beta


C_GRID = (0, 1, 2, 4, 8, 16, 32)


def gate_x(delta, c):
    """Return the rowwise reliability-gated conserved regressor.

    The explicit c=0 branch also defines the zero-energy row without 0/0.
    """
    delta = _matrix(delta, 'donor delta')
    if not np.isfinite(c) or c < 0:
        raise ValueError('Gate c must be finite and nonnegative')
    if c == 0:
        return delta
    energy = np.mean(delta * delta, axis=1)
    return (energy / (energy + c))[:, None] * delta


@dataclass
class GatedTransferModel(TransferModel):
    c: float

    def predict_correction(self, delta, *, groups, trajectory=None):
        delta = _matrix(delta, 'query donor delta')
        groups = _labels(groups, 'query groups', len(delta))
        if set(groups) & set(self.fit_groups):
            raise PermissionError('Transfer query overlaps training chemicals')
        if not set(groups) <= set(self.query_groups):
            raise PermissionError('Transfer query outside declared inventory')
        scores = self.basis.project(delta, trajectory)
        result = (scores[:, :len(self.coefficients)] @ self.coefficients
                  + self.beta * gate_x(delta, self.c))
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite transfer prediction')
        return result


def select_gated_transfers(delta, oofs, *, ids, groups, query_groups, trajectory=None,
                          ranks=RANKS, lambdas=LAMBDAS, c_grid=C_GRID, callback=None):
    """Return the best rank/ridge model for every (family, c), in grid order.

    The caller chooses c by minimum selected_cv_mse, retaining grid order
    for ties.  Inputs never include outer-query features or targets.
    Fold bases, error/regressor cross-products and partial-rank predictions
    are reused; this is algebraically the same candidate search as separate
    weighted ridge fits.  Candidate risks pool chemicals equally, including
    when folds contain different numbers of chemicals or rows.
    """
    delta = _matrix(delta, 'donor delta')
    ids, groups = _inventory(ids, groups)
    query_groups = _labels(query_groups, 'outer-query groups')
    if len(delta) != len(ids) or not oofs:
        raise ValueError('OOF and donor inventories must align')
    if trajectory is not None:
        trajectory = _matrix(trajectory, 'trajectory')
        if len(trajectory) != len(ids):
            raise ValueError('Trajectory rows must align')
    ranks, lambdas, c_grid = tuple(ranks), tuple(lambdas), tuple(c_grid)
    if not ranks or any(k != 'full' and (not isinstance(k, int) or k < 1) for k in ranks):
        raise ValueError('Invalid rank grid')
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError('Invalid ridge grid')
    if (not c_grid or any(not np.isfinite(c) or c < 0 for c in c_grid)
            or len(set(c_grid)) != len(c_grid)):
        raise ValueError('Invalid gate grid')
    c_grid = tuple(float(c) for c in c_grid)
    reference_folds, errors, coverage = None, {}, {}
    for family, oof in oofs.items():
        np.testing.assert_array_equal(oof['ids'], ids)
        np.testing.assert_array_equal(oof['groups'], groups)
        coverage[family] = validate_oof_coverage(
            oof['fold_audits'], ids=ids, groups=groups, query_groups=query_groups)
        mapping = {str(i): int(f['fold']) for f in oof['fold_audits'] for i in f['ids_valid']}
        fold = np.asarray([mapping[i] for i in ids])
        if reference_folds is not None:
            np.testing.assert_array_equal(fold, reference_folds)
        reference_folds = fold
        errors[family] = _matrix(oof['errors'], 'own OOF errors')
        if errors[family].shape != delta.shape:
            raise ValueError('Transfer output must preserve original feature coordinates')
    candidates = [(k, float(lam)) for k in ranks for lam in lambdas]
    keys = [(family, c) for family in errors for c in c_grid]
    risks = {key: np.zeros(len(candidates)) for key in keys}
    fold_records = {key: [] for key in keys}
    regressors = {c: gate_x(delta, c) for c in c_grid}
    for f in np.unique(reference_folds):
        train, valid = reference_folds != f, reference_folds == f
        tt = trajectory[train] if trajectory is not None else None
        tv = trajectory[valid] if trajectory is not None else None
        basis = Basis.fit(delta[train], groups[train], tt)
        zv = basis.project(delta[valid], tv)
        r = len(basis.singular_squared)
        valid_count = len(np.unique(groups[valid]))
        vw = chemical_weights(groups[valid])
        vw = vw / vw.sum() * valid_count
        weighted_scores = (basis.train_scores * basis.weights[:, None]).T
        cross_x = {c: weighted_scores @ regressors[c][train] for c in c_grid}
        bounds = sorted(set([0, r] + [min(k, r) for k in ranks if k != 'full']))
        for family, error in errors.items():
            cross_e = weighted_scores @ error[train]
            for c in c_grid:
                key = (family, c)
                beta = scalar_beta(error[train], regressors[c][train], groups[train])
                cross = cross_e - beta * cross_x[c]
                ev = error[valid] - beta * regressors[c][valid]
                fold_risk = np.empty(len(candidates))
                for li, lam in enumerate(lambdas):
                    coef = cross / (basis.singular_squared[:, None] + lam)
                    prediction = np.zeros_like(ev)
                    cumulative = {0: float(np.sum(vw * np.mean(ev * ev, axis=1)))}
                    for a, z in zip(bounds[:-1], bounds[1:]):
                        prediction = prediction + zv[:, a:z] @ coef[a:z]
                        residual = ev - prediction
                        cumulative[z] = float(np.sum(vw * np.mean(residual * residual, axis=1)))
                    for ki, k in enumerate(ranks):
                        actual = r if k == 'full' else min(k, r)
                        fold_risk[ki * len(lambdas) + li] = cumulative[actual]
                risks[key] += fold_risk
                fold_records[key].append(dict(
                    fold=int(f), training_rows=int(train.sum()), validation_rows=int(valid.sum()),
                    training_groups=sorted(set(groups[train])),
                    validation_groups=sorted(set(groups[valid])),
                    validation_chemical_count=valid_count, available_rank=r, c=c, beta=beta,
                    candidates=[dict(rank=k, lambda_value=lam,
                                     weighted_chemical_error_sum=float(value),
                                     validation_cv_mse=float(value / valid_count))
                                for (k, lam), value in zip(candidates, fold_risk)]))
        if callback:
            callback(dict(stage='gated_transfer_cv_fold_complete', fold=int(f), available_rank=r))
    full = Basis.fit(delta, groups, trajectory)
    full_weighted_scores = (full.train_scores * full.weights[:, None]).T
    chemical_count = len(np.unique(groups))
    output = {}
    for family, c in keys:
        risk = risks[(family, c)] / chemical_count
        best = int(np.argmin(risk))
        requested, lam = candidates[best]
        actual = (len(full.singular_squared) if requested == 'full'
                  else min(requested, len(full.singular_squared)))
        beta = scalar_beta(errors[family], regressors[c], groups)
        # Match the direct residual solve at the selected candidate.
        cross = full_weighted_scores[:actual] @ (errors[family] - beta * regressors[c])
        coefficients = cross / (full.singular_squared[:actual, None] + lam)
        audit = dict(
            family=family, mode='gated_beta_T', c=c, c_grid=list(c_grid),
            requested_rank=requested, actual_rank=actual,
            available_rank=len(full.singular_squared), lambda_value=lam, beta=beta,
            lambda_lower_boundary=lam == min(lambdas), lambda_upper_boundary=lam == max(lambdas),
            full_rank_selected=requested == 'full', selected_cv_mse=float(risk[best]),
            candidates=[dict(rank=k, lambda_value=l, cv_mse=float(v))
                        for (k, l), v in zip(candidates, risk)],
            fold_records=fold_records[(family, c)],
            selection='conditional on fixed base OOF errors; not fully nested stack CV',
            c_selection='caller minimizes selected_cv_mse across c in grid order',
            weights='equal chemicals, equal rows within chemical; mean row weight one',
            gate_formula='e/(e+c), e=mean(delta**2) per row; c=0 gives one',
            gated_component='beta only; PCA and trajectory retain original delta',
            beta_refitted_in_each_training_fold=True,
            fitted_intercept=False, input_standardization=False, pc_whitening=False,
            output_dimensions=delta.shape[1], query_targets_used=False,
            trajectory=trajectory is not None, own_oof_audit=oofs[family]['audit'],
            oof_coverage=coverage[family])
        output[(family, c)] = GatedTransferModel(
            full, coefficients, beta, tuple(sorted(set(groups))),
            tuple(sorted(set(query_groups))), audit, c)
    return output
