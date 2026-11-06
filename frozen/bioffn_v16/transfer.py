"""Fold-local PCA ridge transfer on saved base-model OOF errors.

No dataset loading. Head selection is conditional on fixed base OOF errors,
not fully nested selection of the entire base-plus-transfer stack.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .crosscell import _matrix, _inventory, _labels, validate_oof_coverage


RANKS = (16, 32, 64, 128, 256, 'full')
LAMBDAS = (1e3, 1e4, 1e5, 1e6, 1e7, 1e8)


def chemical_weights(groups):
    """Equal total per chemical, mean weight one (original SSE convention)."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return len(groups) / (len(counts) * counts[inverse])


def scalar_beta(errors, delta, groups):
    w = chemical_weights(groups)
    denominator = np.sum(w[:, None] * delta * delta)
    return float(np.sum(w[:, None] * errors * delta) / denominator) if denominator else 0.


def trajectory_rows(delta, groups, conditions):
    """Repeat each chemical's ordered four-condition donor vector per row."""
    delta = _matrix(delta, 'donor delta')
    conditions = np.asarray(conditions)
    if conditions.shape != (len(delta), 2) or len(groups) != len(delta):
        raise ValueError('Trajectory identities and conditions must align')
    order = ((24, 0), (24, 1), (48, 0), (48, 1))
    out = np.empty((len(delta), 4*delta.shape[1]))
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        lookup = {tuple(conditions[i]): i for i in rows}
        if len(rows) != 4 or set(lookup) != set(order):
            raise ValueError('Exactly four distinct conditions per trajectory required')
        out[rows] = np.concatenate([delta[lookup[c]] for c in order])
    return out


@dataclass
class InputTransform:
    trajectory_center: np.ndarray | None = None
    trajectory_scale: float = 1.

    @classmethod
    def fit(cls, delta, groups, trajectory=None):
        if trajectory is None:
            return cls()
        w = chemical_weights(groups)
        mu = np.average(trajectory, axis=0, weights=w)
        centered = trajectory-mu
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        unit = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
        dc = delta-np.average(delta, axis=0, weights=w)
        uc = unit-np.average(unit, axis=0, weights=w)
        energy = np.sum(w[:, None]*uc*uc)
        scale = np.sqrt(np.sum(w[:, None]*dc*dc)/energy) if energy else 0.
        return cls(mu, float(scale))

    def transform(self, delta, trajectory=None):
        delta = _matrix(delta, 'delta')
        if self.trajectory_center is None:
            if trajectory is not None:
                raise ValueError('Unexpected trajectory for current-row transform')
            return delta
        trajectory = _matrix(trajectory, 'trajectory')
        if len(trajectory) != len(delta) or trajectory.shape[1] != len(self.trajectory_center):
            raise ValueError('Trajectory dimensions differ')
        centered = trajectory-self.trajectory_center
        norm = np.linalg.norm(centered, axis=1, keepdims=True)
        unit = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
        return np.concatenate([delta, self.trajectory_scale*unit], axis=1)


@dataclass
class Basis:
    mean: np.ndarray
    vectors: np.ndarray
    singular_squared: np.ndarray
    weights: np.ndarray
    train_scores: np.ndarray
    transform: InputTransform

    @classmethod
    def fit(cls, delta, groups, trajectory=None):
        transform = InputTransform.fit(delta, groups, trajectory)
        x = transform.transform(delta, trajectory)
        w = chemical_weights(groups)
        mean = np.average(x, axis=0, weights=w)
        xc = x-mean
        _, singular, vt = np.linalg.svd(np.sqrt(w[:, None])*xc, full_matrices=False)
        tol = max(xc.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
        rank = int(np.count_nonzero(singular > tol))
        vectors = vt[:rank].T
        return cls(mean, vectors, singular[:rank]**2, w, xc@vectors, transform)

    def project(self, delta, trajectory=None):
        return (self.transform.transform(delta, trajectory)-self.mean)@self.vectors


@dataclass
class TransferModel:
    basis: Basis
    coefficients: np.ndarray
    beta: float
    fit_groups: tuple
    query_groups: tuple
    audit: dict

    def predict_correction(self, delta, *, groups, trajectory=None):
        delta = _matrix(delta, 'query donor delta')
        groups = _labels(groups, 'query groups', len(delta))
        if set(groups) & set(self.fit_groups):
            raise PermissionError('Transfer query overlaps training chemicals')
        if not set(groups) <= set(self.query_groups):
            raise PermissionError('Transfer query outside declared inventory')
        scores = self.basis.project(delta, trajectory)
        result = scores[:, :len(self.coefficients)]@self.coefficients + self.beta*delta
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite transfer prediction')
        return result

    def original_operator(self):
        """Current-row T only; beta deliberately excluded from its spectrum."""
        if self.basis.transform.trajectory_center is not None:
            raise ValueError('Trajectory transform is not a single linear original-coordinate map')
        return self.basis.vectors[:, :len(self.coefficients)]@self.coefficients


def select_transfers(delta, oofs, *, ids, groups, query_groups, trajectory=None,
                     ranks=RANKS, lambdas=LAMBDAS, include_beta=True, callback=None):
    """Fit identical candidate paths per family's own OOF errors, share bases.

    Returns models indexed by (family, mode), mode='T' or 'beta_T'. Query
    features/targets are not arguments. PCA preprocessing is fold-local.
    """
    delta = _matrix(delta, 'donor delta')
    ids, groups = _inventory(ids, groups)
    if len(delta) != len(ids) or not oofs:
        raise ValueError('OOF and donor inventories must align')
    if trajectory is not None:
        trajectory = _matrix(trajectory, 'trajectory')
        if len(trajectory) != len(ids):
            raise ValueError('Trajectory rows must align')
    if not ranks or any(k != 'full' and (not isinstance(k, int) or k < 1) for k in ranks):
        raise ValueError('Invalid rank grid')
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError('Invalid ridge grid')
    reference_folds = None
    errors = {}
    for family, oof in oofs.items():
        np.testing.assert_array_equal(oof['ids'], ids)
        np.testing.assert_array_equal(oof['groups'], groups)
        validate_oof_coverage(oof['fold_audits'], ids=ids, groups=groups, query_groups=query_groups)
        mapping = {str(i): int(f['fold']) for f in oof['fold_audits'] for i in f['ids_valid']}
        fold = np.array([mapping[i] for i in ids])
        if reference_folds is not None:
            np.testing.assert_array_equal(fold, reference_folds)
        reference_folds = fold
        errors[family] = _matrix(oof['errors'], 'own OOF errors')
        if errors[family].shape != delta.shape:
            raise ValueError('Transfer output must preserve original feature coordinates')
    modes = ('T', 'beta_T') if include_beta else ('T',)
    candidates = [(k, float(lam)) for k in ranks for lam in lambdas]
    keys = [(f, m) for f in errors for m in modes]
    risks = {key: np.zeros(len(candidates)) for key in keys}
    fold_records = []
    for f in np.unique(reference_folds):
        train = reference_folds != f
        valid = ~train
        tt = trajectory[train] if trajectory is not None else None
        tv = trajectory[valid] if trajectory is not None else None
        basis = Basis.fit(delta[train], groups[train], tt)
        zv = basis.project(delta[valid], tv)
        vw = chemical_weights(groups[valid])
        # Accumulate equal chemical risks, not an equal mean of unequal folds.
        vw /= vw.sum()
        vw *= len(np.unique(groups[valid]))
        for family, mode in keys:
            beta = scalar_beta(errors[family][train], delta[train], groups[train]) if mode == 'beta_T' else 0.
            et = errors[family][train]-beta*delta[train]
            ev = errors[family][valid]-beta*delta[valid]
            cross = basis.train_scores.T@(basis.weights[:, None]*et)
            for ci, (k, lam) in enumerate(candidates):
                actual = len(basis.singular_squared) if k == 'full' else min(k, len(basis.singular_squared))
                coef = cross[:actual]/(basis.singular_squared[:actual, None]+lam)
                err = ev-zv[:, :actual]@coef
                risks[(family, mode)][ci] += np.sum(vw*np.mean(err*err, axis=1))
        fold_records.append(dict(fold=int(f), training_rows=int(train.sum()),
            validation_rows=int(valid.sum()), available_rank=len(basis.singular_squared),
            training_groups=sorted(set(groups[train])), validation_groups=sorted(set(groups[valid]))))
        if callback:
            callback(dict(stage='transfer_cv_fold_complete', fold=int(f),
                          available_rank=len(basis.singular_squared)))
    full = Basis.fit(delta, groups, trajectory)
    output = {}
    for family, mode in keys:
        risk = risks[(family, mode)]/len(np.unique(groups))
        best = int(np.argmin(risk))
        requested, lam = candidates[best]
        actual = len(full.singular_squared) if requested == 'full' else min(requested, len(full.singular_squared))
        beta = scalar_beta(errors[family], delta, groups) if mode == 'beta_T' else 0.
        remaining = errors[family]-beta*delta
        cross = full.train_scores.T@(full.weights[:, None]*remaining)
        coef = cross[:actual]/(full.singular_squared[:actual, None]+lam)
        audit = dict(family=family, mode=mode, requested_rank=requested, actual_rank=actual,
            available_rank=len(full.singular_squared), lambda_value=lam, beta=beta,
            lambda_lower_boundary=lam == min(lambdas), lambda_upper_boundary=lam == max(lambdas),
            full_rank_selected=requested == 'full', selected_cv_mse=float(risk[best]),
            candidates=[dict(rank=k, lambda_value=l, cv_mse=float(v)) for (k,l),v in zip(candidates,risk)],
            fold_records=fold_records, selection='conditional on fixed base OOF errors; not fully nested stack CV',
            weights='equal chemicals, equal rows within chemical; mean row weight one',
            fitted_intercept=False, input_standardization=False, pc_whitening=False,
            output_dimensions=delta.shape[1], query_targets_used=False,
            trajectory=trajectory is not None, own_oof_audit=oofs[family]['audit'])
        output[(family,mode)] = TransferModel(full, coef, beta, tuple(sorted(set(groups))),
                                             tuple(sorted(set(query_groups))), audit)
    return output


def operator_summary(model, feature_names):
    """Coordinate-dependent descriptive SVD, preserving paired signs."""
    if len(feature_names) != model.basis.vectors.shape[0]:
        raise ValueError('Feature labels must match original input/output coordinates')
    # V has orthonormal columns, so singular values of V W equal those of W.
    uk, singular, vt = np.linalg.svd(model.coefficients, full_matrices=False)
    left = model.basis.vectors[:, :len(model.coefficients)]@uk
    operator = model.original_operator()
    energy = float(np.sum(singular**2))
    d = len(feature_names)
    compartments = np.array([f.split('_')[0] for f in feature_names])
    kinds = np.array([f.split('_')[1] if '_' in f else 'unknown' for f in feature_names])
    channels = np.array([next((c for c in ('DNA','ER','RNA','AGP','Mito') if c in f), 'none')
                         for f in feature_names])
    def shares(vector, labels):
        return {str(c): float(np.sum(vector[labels == c]**2)) for c in np.unique(labels)}
    desc = dict(operator_energy=energy, actual_rank=model.audit['actual_rank'],
        top_energy={str(k): float(np.sum(singular[:k]**2)/energy) if energy else 0. for k in (1,3,10)},
        diagonal_share=float(np.sum(np.diag(operator)**2)/energy) if energy else 0.,
        identity_projection_share=float(np.trace(operator)**2/d/energy) if energy else 0.,
        mean_diagonal=float(np.trace(operator)/d), singular_values=singular.tolist(),
        feature_baseline={str(c):float(np.mean(kinds == c)) for c in np.unique(kinds)},
        directions=[], interpretation='predictive operator, coordinate-dependent, not causal programs')
    for i in range(min(3,len(singular))):
        a,b = left[:,i],vt[i]
        desc['directions'].append(dict(squared_energy_share=float(singular[i]**2/energy) if energy else 0.,
            signed_input_output_cosine=float(a@b), absolute_input_output_cosine=float(abs(a@b)),
            input_family=shares(a,kinds),output_family=shares(b,kinds),
            input_compartment=shares(a,compartments),output_compartment=shares(b,compartments),
            input_channel=shares(a,channels),output_channel=shares(b,channels),
            input_top_features=[feature_names[j] for j in np.argsort(-abs(a))[:5]],
            output_top_features=[feature_names[j] for j in np.argsort(-abs(b))[:5]]))
    return desc, vt[:min(5,len(singular))].T
