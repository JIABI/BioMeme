"""Optional scalar reference-cell correction on already admitted arrays.

The donor response is the A549 target slots 1:4 mean minus source slot 0,
matched by chemical, time and task before entering this module. Each model
and episode fits one unrestricted, no-intercept scalar from target-cell OOF
errors. This module does not read data, fit a base model or accept query
targets at prediction time.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np


FORMAT = "bioffn_v16_crosscell_calibration_v1"


def _matrix(values, name):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or not all(values.shape)
            or not np.isfinite(values).all()):
        raise ValueError(f"{name} must be a nonempty finite row-by-feature matrix")
    return values


def _labels(values, name, n=None, *, unique=False):
    raw = np.asarray(values)
    if raw.ndim != 1 or (n is not None and len(raw) != n):
        raise ValueError(f"{name} must be aligned one-dimensional labels")
    if any(v is None for v in raw.tolist()):
        raise ValueError(f"{name} contains a missing label")
    labels = raw.astype(str)
    if any(not v.strip() for v in labels):
        raise ValueError(f"{name} contains an empty label")
    if unique and len(set(labels.tolist())) != len(labels):
        raise ValueError(f"{name} must be unique; duplicate IDs are not permitted")
    return labels


def _inventory(ids, groups):
    ids = _labels(ids, "row IDs", unique=True)
    groups = _labels(groups, "chemical groups", len(ids))
    if not len(ids):
        raise ValueError("An admitted training row inventory is required")
    return ids, groups


def _fold_labels(pack):
    required = ("fold", "ids", "groups", "ids_valid")
    if any(key not in pack for key in required):
        raise ValueError("Every OOF fold needs fold, ids, groups and ids_valid")
    fold = pack["fold"]
    if isinstance(fold, (bool, np.bool_)) or not isinstance(fold, (int, np.integer)):
        raise ValueError("OOF fold identifiers must be integers")
    train_ids, train_groups = _inventory(pack["ids"], pack["groups"])
    valid_ids = _labels(pack["ids_valid"], "OOF validation IDs", unique=True)
    if not len(valid_ids):
        raise ValueError("Every OOF fold needs validation rows")
    values = pack.get("valid_groups", pack.get("groups_valid"))
    valid_groups = _labels(values, "OOF validation groups", len(valid_ids))
    if "groups_valid" in pack and "valid_groups" in pack:
        alias = _labels(pack["groups_valid"], "groups_valid", len(valid_ids))
        if not np.array_equal(alias, valid_groups):
            raise ValueError("OOF validation group aliases disagree")
    return int(fold), train_ids, train_groups, valid_ids, valid_groups


def validate_oof_coverage(fold_audits, *, ids, groups, query_groups):
    """Audit chemical-disjoint folds and exactly one OOF value per fit row.

    Accept V16 CV packs, or lightweight records with ``fold``, ``ids``,
    ``groups``, ``ids_valid`` and ``valid_groups`` (``groups_valid`` is an
    accepted alias). Folds may also contain admitted rows outside the beta
    fit inventory, such as source-cell rows. All recorded rows are checked
    for outer-query overlap and consistent identity; every requested fit row
    must occur in validation exactly once. The records describe provenance;
    the caller must supply predictions produced by those actual fold fits.
    """
    ids, groups = _inventory(ids, groups)
    query = set(_labels(query_groups, "outer-query groups").tolist())
    if query & set(groups):
        raise PermissionError("Outer-query chemical cannot enter beta fit")
    expected = dict(zip(ids.tolist(), groups.tolist()))
    identity = dict(expected)
    seen_ids, valid_fold_by_group, folds = set(), {}, []
    for pack in fold_audits:
        fold, ti, tg, vi, vg = _fold_labels(pack)
        if fold in folds:
            raise ValueError("Duplicate OOF fold identifier")
        folds.append(fold)
        if query & (set(tg) | set(vg)):
            raise PermissionError("Outer-query chemical occurs in OOF fold records")
        if set(tg) & set(vg):
            raise PermissionError("A chemical is split across OOF train/validation")
        if seen_ids & set(vi):
            raise ValueError("OOF validation IDs must be covered exactly once")
        seen_ids.update(vi)
        for row_id, group in zip(np.concatenate([ti, vi]), np.concatenate([tg, vg])):
            if row_id in identity and identity[row_id] != group:
                raise ValueError("OOF row ID/group alignment mismatch")
            identity[row_id] = group
        for group in set(vg):
            if group in valid_fold_by_group:
                raise PermissionError("A chemical is split across OOF validation folds")
            valid_fold_by_group[group] = fold
    if not folds:
        raise ValueError("At least one OOF fold audit is required")
    missing = set(expected) - seen_ids
    if missing:
        raise ValueError("Incomplete OOF coverage; missing fit row IDs: "
                         + ", ".join(sorted(missing)))
    return dict(rows=len(ids), chemical_count=len(set(groups)), fold_count=len(folds),
                folds=sorted(folds), fit_row_ids=ids.tolist(),
                fit_groups=sorted(set(groups)), query_groups=sorted(query),
                exact_once_oof_coverage=True, chemical_disjoint_folds=True,
                outer_query_chemicals_excluded=True)


def reconstruct_oof_errors(packs, predictions, *, ids, groups, query_groups):
    """Align ``R_valid - prediction`` from fold packs to the beta-fit IDs.

    Each prediction is a residual prediction in the coordinates of that
    pack's ``R_valid``, in ``ids_valid`` order. Thus subtraction recovers the
    full prediction error without needing a separately reconstructed base
    prediction. Pass one matrix per pack in the same sequence order.
    """
    packs, predictions = list(packs), list(predictions)
    if len(predictions) != len(packs):
        raise ValueError("Need one aligned prediction matrix for every OOF fold")
    ids, groups = _inventory(ids, groups)
    audit = validate_oof_coverage(packs, ids=ids, groups=groups, query_groups=query_groups)
    by_id, fold_audits, feature_dim = {}, [], None
    wanted = set(ids)
    for pack, prediction in zip(packs, predictions):
        fold, ti, tg, vi, vg = _fold_labels(pack)
        if "R_valid" not in pack:
            raise ValueError("OOF packs require R_valid residual targets")
        residual = _matrix(pack["R_valid"], "OOF residual targets")
        prediction = _matrix(prediction, "OOF residual prediction")
        if residual.shape != prediction.shape or len(residual) != len(vi):
            raise ValueError("OOF targets, predictions and row IDs must be aligned")
        if feature_dim is not None and residual.shape[1] != feature_dim:
            raise ValueError("OOF feature dimensions differ between folds")
        feature_dim = residual.shape[1]
        with np.errstate(over="ignore", invalid="ignore"):
            error = residual - prediction
        if not np.isfinite(error).all():
            raise ValueError("OOF errors must be finite")
        by_id.update((row_id, value) for row_id, value in zip(vi, error) if row_id in wanted)
        fold_audits.append(dict(fold=fold, ids=ti.tolist(), groups=tg.tolist(),
                                ids_valid=vi.tolist(), valid_groups=vg.tolist()))
    return dict(errors=np.stack([by_id[row_id] for row_id in ids]), ids=ids,
                groups=groups, fold_audits=fold_audits, audit=audit)


def reference_cell_delta(source_slot0, target_slots123):
    """Return mean of three target replicates minus matched source slot 0.

    Inputs are already matched A549 arrays: source ``(rows, features)`` and
    target ``(rows, 3, features)`` for slots 1, 2 and 3, respectively. Row
    identity and the chemical/time/task matching belong to the data adapter.
    """
    source = _matrix(source_slot0, "A549 source slot 0")
    target = np.asarray(target_slots123, dtype=float)
    if target.shape != (len(source), 3, source.shape[1]) or not np.isfinite(target).all():
        raise ValueError("A549 target slots 1:4 must be finite aligned (rows, 3, features)")
    with np.errstate(over="ignore", invalid="ignore"):
        delta = (target / 3).sum(axis=1) - source
    if not np.isfinite(delta).all():
        raise ValueError("A549 donor delta must be finite")
    return delta


@dataclass(frozen=True)
class CrossCellCalibration:
    beta: float
    feature_dim: int
    fit_groups: tuple
    query_groups: tuple
    audit: dict

    @classmethod
    def fit(cls, oof_errors, donor_delta, *, ids, groups, query_groups, fold_audits):
        """Fit one no-intercept beta; each chemical has equal total weight.

        Both matrices share the supplied row IDs and feature order. Donor
        delta is matched by chemical/time/task in the caller, without using
        target-cell query outcomes. Rows within each chemical have equal
        weight. No clipping, featurewise coefficient or tuning is performed.
        """
        errors = _matrix(oof_errors, "OOF errors")
        delta = _matrix(donor_delta, "donor delta")
        ids, groups = _inventory(ids, groups)
        if errors.shape != delta.shape or len(errors) != len(ids):
            raise ValueError("OOF errors, donor deltas and row IDs must be aligned")
        audit = validate_oof_coverage(fold_audits, ids=ids, groups=groups,
                                      query_groups=query_groups)
        _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = 1. / (len(counts) * counts[inverse])
        donor_scale = float(np.abs(delta).max())
        error_scale = float(np.abs(errors).max())
        numerator = denominator = 0.
        if donor_scale == 0.:
            beta, status = 0., "zero_donor_norm"
        else:
            # Scaling preserves the exact LS objective while avoiding squared
            # overflow/underflow for finite donor coordinates of unusual size.
            d = delta / donor_scale
            e = errors / error_scale if error_scale else errors
            numerator = float(np.sum(weights * np.einsum("ij,ij->i", d, e)))
            denominator = float(np.sum(weights * np.einsum("ij,ij->i", d, d)))
            beta = float((np.longdouble(numerator) / denominator)
                         * np.longdouble(error_scale) / np.longdouble(donor_scale))
            if not np.isfinite(beta):
                raise ValueError("The fitted beta cannot be represented as a finite scalar")
            status = "fitted"
        audit.update(status=status, donor_norm_zero=donor_scale == 0.,
                     pooling="equal chemicals; equal rows within chemical",
                     intercept=False, coefficient_count=1, coefficient_constraint="unrestricted",
                     donor_cell="A549", donor_response="mean(target slots 1:4) - source slot 0",
                     scaled_numerator=numerator, scaled_denominator=denominator,
                     donor_scale=donor_scale, error_scale=error_scale,
                     query_targets_used=False)
        return cls(beta, errors.shape[1], tuple(audit["fit_groups"]),
                   tuple(audit["query_groups"]), audit)

    def predict(self, prediction, donor_delta, *, ids, groups):
        """Add the frozen correction to query predictions; no target input."""
        prediction = _matrix(prediction, "query prediction")
        delta = _matrix(donor_delta, "query donor delta")
        ids, groups = _inventory(ids, groups)
        if (prediction.shape != delta.shape or len(prediction) != len(ids)
                or prediction.shape[1] != self.feature_dim):
            raise ValueError("Query predictions, donor deltas and identities must be aligned")
        if set(groups) & set(self.fit_groups):
            raise PermissionError("Query chemical overlaps beta-fit chemicals")
        if self.query_groups and not set(groups) <= set(self.query_groups):
            raise ValueError("Prediction groups must belong to the declared outer-query inventory")
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = prediction + self.beta * delta
        if not np.isfinite(corrected).all():
            raise ValueError("Corrected predictions must be finite")
        return corrected

    def to_dict(self):
        return dict(format=FORMAT, beta=self.beta, feature_dim=self.feature_dim,
                    fit_groups=list(self.fit_groups), query_groups=list(self.query_groups),
                    audit=copy.deepcopy(self.audit))

    @classmethod
    def from_dict(cls, value):
        if value.get("format") != FORMAT:
            raise ValueError("Unknown cross-cell calibration artifact")
        beta, d = value.get("beta"), value.get("feature_dim")
        if (isinstance(beta, bool) or not isinstance(beta, (int, float))
                or not np.isfinite(beta)):
            raise ValueError("Artifact beta must be a finite scalar")
        if isinstance(d, bool) or not isinstance(d, int) or d < 1:
            raise ValueError("Artifact feature dimension must be a positive integer")
        fit = _labels(value.get("fit_groups", []), "fit groups", unique=True)
        query = _labels(value.get("query_groups", []), "query groups", unique=True)
        if not len(fit):
            raise ValueError("Artifact requires beta-fit chemical provenance")
        if set(fit) & set(query):
            raise PermissionError("Artifact fit/query chemical inventories overlap")
        return cls(float(beta), d, tuple(fit), tuple(query), copy.deepcopy(value.get("audit", {})))
