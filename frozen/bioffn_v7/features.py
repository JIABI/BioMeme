"""FIT-only coordinates for the small conditional residual model.

Input PCA preserves the retained covariance spectrum: each score block has one
scalar RMS normalization, never a separate scale for each principal component.
Signed prior scores use centered raw RNA. Output directions are uncentered
residual PCs confined to predefined feature groups; the full prediction remains
the uncompressed baseline plus a correction in this fixed subspace.
"""
from dataclasses import dataclass
import copy
import math

import numpy as np


_EPS = 1e-12


def coarsen_feature_groups(feature_names):
    """Return fixed compartment/family IDs, dropping channel subdivisions.

    For CellProfiler names such as ``Cells_Texture_Contrast_DNA_3_00``, the
    group is ``Cells/Texture``. Labels are sorted, so assignments do not depend
    on the order in which groups happen to appear. No observed values are used.
    """
    names = list(feature_names)
    if not names:
        raise ValueError("feature_names must not be empty")
    labels = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError("feature names must be strings")
        parts = name.split("_")
        if len(parts) < 2 or not all(parts[:2]):
            raise ValueError("Feature names must contain compartment and family")
        labels.append("/".join(parts[:2]))
    unique = sorted(set(labels))
    index = {label: i for i, label in enumerate(unique)}
    return np.asarray([index[label] for label in labels], dtype=np.int64), unique


def _matrix(value, name, *, rows=None, columns=None):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty-width matrix")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count differs from FIT inputs")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} column count differs from the fitted feature map")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _rank(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _chemical_weights(groups, rows):
    labels = np.asarray(groups)
    if labels.ndim != 1 or len(labels) != rows:
        raise ValueError("groups must assign every FIT row to one chemical")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError("groups contains non-finite labels")
    if labels.dtype.kind == "O" and any(value is None for value in labels):
        raise ValueError("groups contains missing labels")
    try:
        unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    except TypeError as error:
        raise ValueError("groups must have consistently typed labels") from error
    return 1. / (len(unique) * counts[inverse]), len(unique)


def _standardizer(values, weights):
    mean = weights @ values
    variance = weights @ np.square(values - mean)
    scale = np.sqrt(np.maximum(variance, 0.))
    constant = scale <= _EPS
    scale = np.where(constant, 1., scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("FIT standardization produced non-finite moments")
    return mean, scale, constant


def _canonical_columns(matrix):
    """Fix SVD sign ambiguity without rotating or whitening directions."""
    matrix = matrix.copy()
    for column in range(matrix.shape[1]):
        pivot = int(np.argmax(np.abs(matrix[:, column])))
        if matrix[pivot, column] < 0:
            matrix[:, column] *= -1
    return matrix


def _numerical_rank(singular, shape):
    if not len(singular) or singular[0] <= _EPS:
        return 0
    tolerance = max(shape) * np.finfo(np.float64).eps * singular[0]
    return int(np.sum(singular > tolerance))


def _pca(values, weights, requested_rank):
    mean, scale, constant = _standardizer(values, weights)
    standardized = (values - mean) / scale
    weighted = standardized * np.sqrt(weights[:, None])
    _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    # Dimensional capacity depends on sample/feature counts, not an observed
    # eigenvalue threshold. Rank-deficient retained coordinates stay explicit.
    rank = min(requested_rank, values.shape[1], values.shape[0] - 1)
    components = _canonical_columns(vt[:rank].T)
    scores = standardized @ components
    score_energy = float(weights @ np.square(scores).sum(1))
    rms = math.sqrt(max(0., score_energy / rank))
    block_scale = rms if rms > _EPS else 1.
    total_energy = float(np.square(weighted).sum())
    audit = dict(requested_rank=requested_rank, retained_rank=rank,
                 numerical_rank=_numerical_rank(singular, weighted.shape),
                 constant_input_columns=int(constant.sum()),
                 retained_energy_fraction=(score_energy / total_energy if total_energy > _EPS else 0.),
                 retained_score_variances=(weights @ np.square(scores)).tolist(),
                 block_rms_scale=block_scale,
                 normalization="one scalar for the entire retained PC block; no per-PC whitening")
    return mean, scale, components, block_scale, audit


def _output_basis(residual, weights, feature_groups, rank_per_group):
    groups = np.asarray(feature_groups)
    if groups.ndim != 1 or len(groups) != residual.shape[1]:
        raise ValueError("feature_groups must assign every output feature")
    if groups.dtype.kind not in "iu" or np.any(groups < 0):
        raise ValueError("feature_groups must be nonnegative integer IDs")
    columns, basis_groups, records = [], [], []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        weighted = residual[:, indices] * np.sqrt(weights[:, None])
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        rank = min(rank_per_group, len(indices), len(residual))
        local = _canonical_columns(vt[:rank].T)
        block = np.zeros((residual.shape[1], rank), dtype=np.float64)
        block[indices] = local
        columns.append(block)
        basis_groups.extend([int(group)] * rank)
        energy = float(np.square(weighted).sum())
        retained = float(weights @ np.square(residual[:, indices] @ local).sum(1))
        records.append(dict(group=int(group), feature_count=len(indices), retained_rank=rank,
                            numerical_rank=_numerical_rank(singular, weighted.shape),
                            residual_energy=energy, retained_energy=retained,
                            retained_energy_fraction=retained / energy if energy > _EPS else 0.))
    basis = np.concatenate(columns, axis=1)
    coefficients = residual @ basis
    retained = float(weights @ np.square(coefficients).sum(1))
    total = float(weights @ np.square(residual).sum(1))
    scale = math.sqrt(max(0., retained / basis.shape[1]))
    output_scale = scale if scale > _EPS else 1.
    audit = dict(rank_per_group=rank_per_group, groups=len(records), rank=basis.shape[1],
                 feature_groups=records, centered=False,
                 retained_energy_fraction=retained / total if total > _EPS else 0.,
                 total_residual_energy=total, retained_residual_energy=retained,
                 output_scale=output_scale,
                 scale_definition="sqrt(chemical-weighted mean sum(projected residual**2) / basis rank)",
                 scope="fixed group-supported correction basis; the full baseline output is not compressed")
    return basis.astype(np.float32), groups.astype(np.int64), np.asarray(basis_groups, np.int64), output_scale, audit


@dataclass(frozen=True)
class FeatureMap:
    x_mean: np.ndarray
    x_scale: np.ndarray
    x_components: np.ndarray
    x_block_scale: float
    rna_mean: np.ndarray
    rna_scale: np.ndarray
    rna_components: np.ndarray
    rna_block_scale: float
    program_prior: np.ndarray
    program_mean: np.ndarray
    program_scale: np.ndarray
    condition_mean: np.ndarray
    condition_scale: np.ndarray
    basis: np.ndarray
    output_scale: float
    feature_groups: np.ndarray
    basis_groups: np.ndarray
    audit: dict

    @classmethod
    def fit(cls, x, rna, conditions, prior, groups, residual, feature_groups,
            x_rank=16, rna_rank=16, output_rank_per_group=1):
        """Fit on source FIT arrays and honest FIT OOF residuals only.

        This array-level object has no evaluation-data argument. The caller
        enforces FIT roles and the residuals' OOF lineage; their provenance
        cannot be inferred from numerical arrays.
        """
        x = _matrix(x, "x")
        if len(x) < 2:
            raise ValueError("At least two FIT rows are required")
        rna = _matrix(rna, "rna", rows=len(x))
        conditions = _matrix(conditions, "conditions", rows=len(x))
        residual = _matrix(residual, "residual", rows=len(x), columns=x.shape[1])
        prior = _matrix(prior, "prior", rows=rna.shape[1])
        x_rank, rna_rank = _rank(x_rank, "x_rank"), _rank(rna_rank, "rna_rank")
        output_rank_per_group = _rank(output_rank_per_group, "output_rank_per_group")
        weights, chemical_count = _chemical_weights(groups, len(x))
        xm, xs, xp, xrms, xa = _pca(x, weights, x_rank)
        rm, rs, rp, rrms, ra = _pca(rna, weights, rna_rank)
        norms = np.linalg.norm(prior, axis=0)
        if not np.isfinite(norms).all():
            raise ValueError("Prior column norm is non-finite")
        active_prior = norms > _EPS
        normalized_prior = prior / np.where(active_prior, norms, 1.)
        # Centered RAW RNA preserves the meaning/sign of the supplied footprint.
        # This is deliberately different from the gene-standardized PCA path.
        raw_programs = (rna - rm) @ normalized_prior
        pm, ps, constant_programs = _standardizer(raw_programs, weights)
        cm, cs, constant_conditions = _standardizer(conditions, weights)
        basis, feature_groups, basis_groups, output_scale, oa = _output_basis(
            residual, weights, feature_groups, output_rank_per_group)
        audit = dict(
            version="bioffn_v7_feature_map_v1", fit_rows=len(x), fit_chemical_groups=chemical_count,
            sample_weighting="each chemical has equal total weight; its FIT rows share that weight",
            input_pca=dict(x=xa, rna=ra), output_basis=oa,
            prior=dict(programs=prior.shape[1], gene_count=prior.shape[0],
                       zero_columns=int((~active_prior).sum()),
                       positive_weights=int((prior > 0).sum()), negative_weights=int((prior < 0).sum()),
                       constant_programs=int(constant_programs.sum()),
                       projection="FIT-centered raw RNA times column-L2-normalized signed prior",
                       score_scaling="FIT per-program mean/std; no absolute-value transform"),
            constant_conditions=int(constant_conditions.sum()),
            dimensions=dict(x_input=x.shape[1], rna_input=rna.shape[1], conditions=conditions.shape[1],
                            programs=prior.shape[1], x_rank=xp.shape[1], rna_rank=rp.shape[1],
                            output_features=residual.shape[1], output_rank=basis.shape[1]),
            fitting_scope="caller-supplied FIT arrays only; no evaluation arguments",
            residual_provenance="caller must supply honest OOF FIT residuals; not inferred from arrays",
            decode="normalized coefficients * output_scale @ basis.T; no residual mean added",
        )
        return cls(xm, xs, xp, xrms, rm, rs, rp, rrms, normalized_prior, pm, ps,
                   cm, cs, basis, output_scale, feature_groups, basis_groups, audit)

    def transform(self, x, rna, conditions):
        """Apply immutable FIT coordinates independently to every query row."""
        x = _matrix(x, "x", columns=len(self.x_mean))
        rna = _matrix(rna, "rna", rows=len(x), columns=len(self.rna_mean))
        conditions = _matrix(conditions, "conditions", rows=len(x), columns=len(self.condition_mean))
        values = dict(
            x=((x - self.x_mean) / self.x_scale) @ self.x_components / self.x_block_scale,
            r=((rna - self.rna_mean) / self.rna_scale) @ self.rna_components / self.rna_block_scale,
            p=(((rna - self.rna_mean) @ self.program_prior) - self.program_mean) / self.program_scale,
            c=(conditions - self.condition_mean) / self.condition_scale,
        )
        output = {key: value.astype(np.float32) for key, value in values.items()}
        if any(not np.isfinite(value).all() for value in output.values()):
            raise ValueError("Transformed query coordinates are not finite float32 values")
        return output

    def to_dict(self):
        result = {name: value.tolist() if isinstance(value, np.ndarray) else copy.deepcopy(value)
                  for name, value in vars(self).items()}
        result["format"] = "bioffn_v7_feature_map_v1"
        return result

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("format") != "bioffn_v7_feature_map_v1":
            raise ValueError("Unknown feature-map format")
        data = copy.deepcopy(value)
        data.pop("format")
        arrays = ("x_mean", "x_scale", "x_components", "rna_mean", "rna_scale", "rna_components",
                  "program_prior", "program_mean", "program_scale", "condition_mean", "condition_scale",
                  "basis", "feature_groups", "basis_groups")
        for name in arrays:
            dtype = np.int64 if name in {"feature_groups", "basis_groups"} else np.float32 if name == "basis" else np.float64
            data[name] = np.asarray(data[name], dtype=dtype)
            if not np.isfinite(data[name]).all():
                raise ValueError("Feature map contains non-finite values")
        for name in ("x_block_scale", "rna_block_scale", "output_scale"):
            data[name] = float(data[name])
            if not math.isfinite(data[name]) or data[name] <= 0:
                raise ValueError("Feature-map scalar scales must be finite and positive")
        for name in ("x_scale", "rna_scale", "program_scale", "condition_scale"):
            if data[name].ndim != 1 or np.any(data[name] <= 0):
                raise ValueError("Feature-map coordinate scales must be positive vectors")
        obj = cls(**data)
        d, g, c, m = len(obj.x_mean), len(obj.rna_mean), len(obj.condition_mean), len(obj.program_mean)
        if (obj.x_mean.ndim != 1 or obj.rna_mean.ndim != 1 or obj.condition_mean.ndim != 1
                or obj.x_scale.shape != (d,) or obj.rna_scale.shape != (g,)
                or obj.condition_scale.shape != (c,) or obj.program_scale.shape != (m,)
                or obj.program_prior.shape != (g, m)
                or obj.x_components.ndim != 2 or obj.x_components.shape[0] != d
                or obj.rna_components.ndim != 2 or obj.rna_components.shape[0] != g
                or obj.basis.ndim != 2 or obj.basis.shape[0] != d
                or obj.feature_groups.shape != (d,) or obj.basis_groups.shape != (obj.basis.shape[1],)):
            raise ValueError("Feature-map dimensions are inconsistent")
        return obj
