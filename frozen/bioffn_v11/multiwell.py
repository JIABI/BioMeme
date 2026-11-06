"""Condition-specific multi-source scalar calibration and input-only routing.

G_c = mu_y,c + sum_j b_c,j (x_j - mu_c,j).
There is one unconstrained coefficient per active stream, shared across all
output features. Chemical-equal weighted least squares uses every coordinate;
rank-deficient designs take the minimum-norm solution. Nothing is standardized
using query outcomes. Stream construction never accesses ``data.target``.
"""
from __future__ import annotations

import copy
import numpy as np

from bioffn_v7 import baselines as BL


ROUTES = ("same", "other", "low", "low_other")
MODES = ("single", "cross_time", "same_time_lower", "all_slot0", "available_time")


def _stream_arrays(streams, active, codes):
    streams = np.asarray(streams, dtype=float)
    active = np.asarray(active)
    if streams.ndim != 3 or streams.shape[1] != 4 or min(streams.shape) < 1:
        raise ValueError("Streams must have shape [rows,4,features]")
    if active.shape != streams.shape[:2] or active.dtype != bool:
        raise ValueError("Active routes must be boolean [rows,4]")
    if not active.any(axis=1).all() or not np.isfinite(streams[active]).all():
        raise ValueError("Every row needs at least one finite active input stream")
    codes = BL._labels(codes, "codes", len(streams))
    for code in np.unique(codes):
        patterns = active[codes == code]
        if not np.all(patterns == patterns[0]):
            raise ValueError("Active pattern must be constant within each condition")
    return streams, active, codes


class MultiwellCalibration:
    def fit(self, streams, active, y, codes, groups):
        streams, active, codes = _stream_arrays(streams, active, codes)
        y = BL._matrix(y, "target", len(streams))
        groups = BL._labels(groups, "groups", len(streams))
        if y.shape[1] != streams.shape[2]:
            raise ValueError("Source and output feature dimensions differ")
        self.n_features = y.shape[1]
        self.source_mean, self.target_mean, self.coefficients, self.active_patterns = {}, {}, {}, {}
        conditions = {}
        for code in np.unique(codes):
            mask = codes == code
            pattern = active[mask][0].copy()
            routes = np.flatnonzero(pattern)
            x, target = streams[mask][:, routes, :], y[mask]
            w = BL.group_weights(groups[mask])
            mean = np.einsum("n,nkd->kd", w, x)
            target_mean = w@target
            # Preserve old scalar calibration's operation order exactly.
            if len(routes) == 1:
                mean[0] = w@x[:, 0]
            xc, yc = x-mean, target-target_mean
            sw = len(w)*w
            if len(routes) == 1:
                xx = sw@(xc[:, 0]*xc[:, 0])
                xy = sw@(xc[:, 0]*yc)
                denominator = float(xx.sum())
                fitted = np.array([float(xy.sum()/denominator) if denominator > 1e-12 else 0.])
                rank = int(denominator > 1e-12)
                singular = np.array([np.sqrt(max(denominator, 0.))])
                solver = "exact V7 scalar formula; denominator floor 1e-12"
            else:
                root_w = np.sqrt(sw)
                design = (root_w[:, None, None]*xc).transpose(0, 2, 1).reshape(-1, len(routes))
                response = (root_w[:, None]*yc).reshape(-1)
                fitted, _, rank, singular = np.linalg.lstsq(design, response, rcond=None)
                solver = "full-feature weighted least squares; SVD minimum norm"
            means = np.zeros((4, self.n_features))
            coefficients = np.zeros(4)
            means[routes], coefficients[routes] = mean, fitted
            self.source_mean[code], self.target_mean[code] = means, target_mean
            self.coefficients[code], self.active_patterns[code] = coefficients, pattern
            conditions[str(code)] = {"rows": int(mask.sum()), "groups": len(np.unique(groups[mask])),
                "fit_groups": np.unique(groups[mask]).tolist(), "active_routes": [ROUTES[j] for j in routes],
                "coefficients": coefficients.tolist(), "design_rank": int(rank),
                "design_singular_values": singular.tolist(), "solver": solver,
                "chemical_weight_sum": float(w.sum()), "weighted_sum_normalization": float(sw.sum()),
                "source_mean_rms": np.sqrt(np.mean(means*means, axis=1)).tolist(),
                "target_mean_rms": float(np.sqrt(np.mean(target_mean*target_mean)))}
        self.audit = {"method": "condition-specific chemical-equal multi-stream scalar OLS",
            "formula": "mu_y + sum_j b_j*(x_j-mu_j)", "fit_rows": len(streams),
            "fit_groups": np.unique(groups).tolist(), "output_dim": self.n_features,
            "coefficients_unconstrained": True, "per_feature_coefficients": False,
            "feature_standardization": "none; original feature units", "conditions": conditions}
        return self

    def predict(self, streams, active, codes):
        if not hasattr(self, "n_features"):
            raise ValueError("Calibration has not been fitted")
        streams, active, codes = _stream_arrays(streams, active, codes)
        if streams.shape[2] != self.n_features:
            raise ValueError("Source feature dimension differs from fitted calibration")
        if set(codes)-set(self.source_mean):
            raise ValueError("Prediction contains unseen conditions")
        result = np.empty((len(streams), self.n_features))
        for code in np.unique(codes):
            mask = codes == code
            if not np.array_equal(active[mask][0], self.active_patterns[code]):
                raise ValueError("Prediction active routes differ from fitted condition")
            routes = np.flatnonzero(self.active_patterns[code])
            if len(routes) == 1:
                j = routes[0]
                result[mask] = self.target_mean[code]+self.coefficients[code][j]*(streams[mask, j]-self.source_mean[code][j])
            else:
                centered = streams[mask][:, routes, :]-self.source_mean[code][routes]
                result[mask] = self.target_mean[code]+np.einsum("j,njd->nd", self.coefficients[code][routes], centered)
        return result

    def to_dict(self):
        if not hasattr(self, "n_features"):
            raise ValueError("Calibration has not been fitted")
        return {"format": "v11_multiwell_calibration", "n_features": self.n_features,
            **{name: {str(k): v.tolist() for k, v in getattr(self, name).items()}
               for name in ("source_mean", "target_mean", "coefficients", "active_patterns")},
            "audit": copy.deepcopy(self.audit)}

    @classmethod
    def from_dict(cls, saved):
        if saved.get("format") != "v11_multiwell_calibration":
            raise ValueError("Unknown multiwell calibration format")
        result = cls()
        result.n_features = int(saved["n_features"])
        if result.n_features < 1:
            raise ValueError("Invalid saved feature dimension")
        for name in ("source_mean", "target_mean", "coefficients", "active_patterns"):
            dtype = bool if name == "active_patterns" else float
            setattr(result, name, {str(k): np.asarray(v, dtype=dtype) for k, v in saved[name].items()})
        keys = set(result.source_mean)
        if not keys or any(set(getattr(result, k)) != keys for k in ("target_mean", "coefficients", "active_patterns")):
            raise ValueError("Saved condition inventories disagree")
        for code in keys:
            fields = (result.source_mean[code], result.target_mean[code], result.coefficients[code])
            if ([v.shape for v in fields] != [(4, result.n_features), (result.n_features,), (4,)]
                    or not all(np.isfinite(v).all() for v in fields)
                    or result.active_patterns[code].shape != (4,) or not result.active_patterns[code].any()):
                raise ValueError("Invalid saved calibration arrays")
            if np.any(result.coefficients[code][~result.active_patterns[code]] != 0):
                raise ValueError("Inactive routes cannot carry coefficients")
        result.audit = copy.deepcopy(saved["audit"])
        return result


def crossfit_multiwell(streams, active, y, codes, groups, n_folds=8, seed=20261008+200009):
    """Return whole-chemical OOF predictions; every fold re-estimates all means."""
    streams, active, codes = _stream_arrays(streams, active, codes)
    y = BL._matrix(y, "target", len(streams))
    groups = BL._labels(groups, "groups", len(streams))
    fid = BL.make_folds(groups, n_folds, seed)
    prediction, audits = np.empty_like(y), []
    for fold in np.unique(fid):
        train, valid = fid != fold, fid == fold
        model = MultiwellCalibration().fit(streams[train], active[train], y[train], codes[train], groups[train])
        prediction[valid] = model.predict(streams[valid], active[valid], codes[valid])
        audits.append({"fold": int(fold), "train_indices": np.flatnonzero(train).tolist(),
            "validation_indices": np.flatnonzero(valid).tolist(),
            "train_groups": np.unique(groups[train]).tolist(), "validation_groups": np.unique(groups[valid]).tolist(),
            "baseline": model.audit})
    return prediction, {"folds": audits, "actual_folds": len(audits), "seed": int(seed),
        "all_rows_predicted_once": True, "group_leakage": False,
        "means_and_coefficients_fit_on_training_fold_only": True}


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing/invalid inventory {key}") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Inventory {key} must be positive and finite")
    return value


def _source_time(row):
    names = [name for name in ("source_morphology_time", "source_profile_time") if name in row]
    if not names:
        raise ValueError("Inventory needs actual source morphology/profile time")
    values = [_number(row, name) for name in names]
    if not np.allclose(values, values[0], atol=1e-10, rtol=0.):
        raise ValueError("Conflicting actual source time metadata")
    return values[0]


def build_streams(data, mode):
    """Route existing slot-0 inputs only, never reading target values.

    The lookup key is (chemical group, target CP time, dose-transfer task).
    Required counterpart rows must share group, layout, role and cell. The
    available-time mode additionally excludes actual source profiles later
    than the query target time. ``coordinate_x`` averages only active
    same-dose routes (same/other), never lower-dose routes.
    """
    if mode not in MODES:
        raise ValueError("Unknown multiwell stream mode")
    x = BL._matrix(data.x, "source profiles")
    n = len(x)
    groups = BL._labels(data.groups, "groups", n)
    ids = BL._labels(data.ids, "ids", n)
    layouts = BL._labels(data.layouts, "layouts", n)
    roles = BL._labels(data.roles, "roles", n)
    inventory = data.metadata.get("row_inventory")
    if not isinstance(inventory, list) or len(inventory) != n or len(np.unique(ids)) != n:
        raise ValueError("Complete, unique and aligned input row inventory is required")
    lookup, times, tasks, source_times, doses, target_doses = {}, [], [], [], [], []
    for i, row in enumerate(inventory):
        for field, expected in (("id", ids[i]), ("group", groups[i]), ("layout", layouts[i]), ("role", roles[i])):
            if str(row.get(field, "")) != str(expected):
                raise ValueError(f"Input inventory {field} does not align with its array row")
        cp = _number(row, "target_time")
        task = row.get("task")
        if cp not in (24., 48.) or isinstance(task, (bool, np.bool_)) or task not in (0, 1):
            raise ValueError("Stream inventory supports CP24/48 and task0/1 only")
        source, target = _number(row, "source_nominal_dose"), _number(row, "target_nominal_dose")
        if not source < target:
            raise ValueError("Every source input nominal dose must be strictly below its target")
        st = _source_time(row)
        key = (str(groups[i]), cp, int(task))
        if key in lookup:
            raise ValueError("Ambiguous duplicate chemical/time/task input identity")
        lookup[key] = i
        times.append(cp); tasks.append(int(task)); source_times.append(st); doses.append(source); target_doses.append(target)
    active = np.zeros((n, 4), dtype=bool)
    indices = np.full((n, 4), -1, dtype=int)
    streams = np.zeros((n, 4, x.shape[1]), dtype=float)
    lineage = []
    for i, row in enumerate(inventory):
        cp, task = times[i], tasks[i]
        other = 48. if cp == 24. else 24.
        wanted = [True, mode in ("cross_time", "all_slot0", "available_time"),
                  task == 1 and mode in ("same_time_lower", "all_slot0", "available_time"),
                  task == 1 and mode in ("all_slot0", "available_time")]
        keys = [(str(groups[i]), cp, task), (str(groups[i]), other, task),
                (str(groups[i]), cp, 0), (str(groups[i]), other, 0)]
        used = []
        for route, requested in enumerate(wanted):
            if not requested:
                continue
            key = keys[route]
            if key not in lookup:
                # An unavailable future-time row is not a required input.
                if mode == "available_time" and key[1] > cp:
                    continue
                raise ValueError(f"Missing required {ROUTES[route]} input identity {key}")
            j = lookup[key]
            if mode == "available_time" and source_times[j] > cp:
                continue
            if (groups[j] != groups[i] or layouts[j] != layouts[i] or roles[j] != roles[i]
                    or inventory[j].get("cell") != row.get("cell")):
                raise ValueError("Cross-row input identity/context does not match the query")
            if not doses[j] < target_doses[i]:
                raise ValueError("An input route is not strictly below the query target dose")
            if route < 2 and not np.isclose(doses[j], doses[i], rtol=1e-8, atol=1e-12):
                raise ValueError("Same-dose time routes have mismatched nominal dose")
            if route >= 2 and not doses[j] < doses[i]:
                raise ValueError("Lower-dose route is not lower than the main source dose")
            active[i, route], indices[i, route], streams[i, route] = True, j, x[j]
            used.append({"route": ROUTES[route], "input_row_id": str(ids[j]), "input_row_index": int(j),
                "source_profile_time": source_times[j], "source_nominal_dose": doses[j]})
        if not active[i, 0]:
            raise ValueError("The query's own source profile is not available at target time")
        lineage.append({"query_id": str(ids[i]), "group": str(groups[i]),
            "target_time": cp, "target_nominal_dose": target_doses[i], "inputs": used})
    codes = np.array([f"CP{int(cp)}|task{task}" for cp, task in zip(times, tasks)])
    _stream_arrays(streams, active, codes)
    coordinate_x = np.sum(streams[:, :2], axis=1)/np.sum(active[:, :2], axis=1)[:, None]
    audit = {"mode": mode, "route_names": list(ROUTES), "rows": n,
        "groups": len(np.unique(groups)), "input_row_lineage": lineage,
        "targets_accessed": False, "rna_accessed": False,
        "coordinate_x": "mean of active same-dose routes only; never lower-dose inputs",
        "availability_rule": "actual source profile time <= target time" if mode == "available_time" else "retrospective inputs; future-time source may be allowed",
        "all_input_doses_below_target": True, "physical_well_disjointness": "runner must audit physical wells separately",
        "condition_active_patterns": {str(code): active[codes == code][0].tolist() for code in np.unique(codes)}}
    return {"streams": streams, "active": active, "coordinate_x": coordinate_x,
            "indices": indices, "audit": audit}
