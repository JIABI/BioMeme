"""Separate reconstruction, direction and identification evaluation.

Readout transforms and centers are supplied from training data. This module
never estimates them from evaluation targets and never selects the best metric.
"""
from __future__ import annotations

import numpy as np


def _matrix(a, name):
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite and two dimensional")
    return a


def cosine_loss(prediction, target):
    pred, target = _matrix(prediction, "prediction"), _matrix(target, "target")
    if pred.shape != target.shape:
        raise ValueError("Prediction and target dimensions differ")
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    cosine = np.divide(np.sum(pred * target, axis=1), denom,
                       out=np.zeros(len(pred)), where=denom > 0)
    return 1 - np.clip(cosine, -1, 1)


def row_metrics(prediction, target, center=None):
    pred, target = _matrix(prediction, "prediction"), _matrix(target, "target")
    if pred.shape != target.shape:
        raise ValueError("Prediction and target dimensions differ")
    a, b = np.linalg.norm(pred, axis=1), np.linalg.norm(target, axis=1)
    d = pred.shape[1]
    direction = cosine_loss(pred, target)
    result = dict(mse=np.mean((pred - target) ** 2, axis=1),
                  norm_error=(a - b) ** 2 / d,
                  weighted_angular=2 * a * b * direction / d,
                  cosine_loss=direction)
    if center is not None:
        center = np.asarray(center, dtype=float)
        if center.shape not in {(d,), pred.shape} or not np.isfinite(center).all():
            raise ValueError("Center must be a frozen feature vector or aligned condition centers")
        result["centered_cosine_loss"] = cosine_loss(pred - center, target - center)
    return result


def group_mean(values, groups):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups).astype(str)
    if (values.ndim != 1 or groups.ndim != 1 or values.shape != groups.shape
            or not len(groups) or not np.isfinite(values).all()):
        raise ValueError("Finite row values and chemical groups must align")
    return float(np.mean([values[groups == g].mean() for g in np.unique(groups)]))


def summarize(prediction, target, groups, center=None):
    return {key: group_mean(value, groups)
            for key, value in row_metrics(prediction, target, center).items()}


def retrieval(prediction, gallery, query_groups, gallery_groups, *, center,
              transform=None, allowed=None):
    """MRR/AP/top1 with average ranks for ties and declared candidate masks.

    ``transform`` is a FIT-only right-multiplication noise readout, [D,D] or
    diagonal [D]. ``allowed`` restricts candidates to the prespecified same
    layout/batch/target condition. Gallery size is reported per query.
    A tied score is assigned its average rank, not arbitrary array order.
    """
    pred, gallery = _matrix(prediction, "prediction"), _matrix(gallery, "gallery")
    qg, gg = np.asarray(query_groups).astype(str), np.asarray(gallery_groups).astype(str)
    if pred.shape[1] != gallery.shape[1] or qg.shape != (len(pred),) or gg.shape != (len(gallery),):
        raise ValueError("Retrieval identities/coordinates do not align")
    center = np.asarray(center, dtype=float)
    if center.shape != (pred.shape[1],) or not np.isfinite(center).all():
        raise ValueError("Retrieval requires an explicit frozen common center")
    pred, gallery = pred - center, gallery - center
    if transform is not None:
        transform = np.asarray(transform, dtype=float)
        d = pred.shape[1]
        if transform.shape == (d,):
            pred, gallery = pred * transform, gallery * transform
        elif transform.shape == (d, d):
            pred, gallery = pred @ transform, gallery @ transform
        else:
            raise ValueError("Readout must be diagonal [D] or full [D,D]")
        if not np.isfinite(transform).all():
            raise ValueError("Readout must be finite")
    pred = np.divide(pred, np.linalg.norm(pred, axis=1, keepdims=True),
                     out=np.zeros_like(pred), where=np.linalg.norm(pred, axis=1, keepdims=True) > 0)
    gallery = np.divide(gallery, np.linalg.norm(gallery, axis=1, keepdims=True),
                        out=np.zeros_like(gallery), where=np.linalg.norm(gallery, axis=1, keepdims=True) > 0)
    scores = pred @ gallery.T
    if allowed is None:
        allowed = np.ones(scores.shape, dtype=bool)
    allowed = np.asarray(allowed)
    if allowed.shape != scores.shape or allowed.dtype != bool:
        raise ValueError("Candidate mask must be boolean [queries,gallery]")
    result = {key: [] for key in ("mrr", "ap", "top1", "gallery_size")}
    for i, score in enumerate(scores):
        values = score[allowed[i]]
        positive = gg[allowed[i]] == qg[i]
        if not positive.any():
            raise ValueError("Every query needs a permitted positive gallery target")
        ranks = 1 + (values[None, :] > values[:, None]).sum(1)
        ranks = ranks + ((values[None, :] == values[:, None]).sum(1) - 1) / 2
        positive_ranks = np.sort(ranks[positive])
        # AP uses expected positive count at tied midrank. This is a declared
        # midrank convention, not exact random-tie expected AP.
        precision = []
        for value, rank in zip(values[positive], ranks[positive]):
            greater_pos = ((values > value) & positive).sum()
            tied_pos = ((values == value) & positive).sum()
            precision.append((greater_pos + (tied_pos + 1) / 2) / rank)
        top = values == values.max()
        result["mrr"].append(1 / positive_ranks[0])
        result["ap"].append(float(np.mean(precision)))
        result["top1"].append(float(positive[top].mean()))
        result["gallery_size"].append(len(values))
    return {key: np.asarray(value) for key, value in result.items()}
