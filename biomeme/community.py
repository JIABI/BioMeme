"""Public prepared-array interface to the unchanged BioMeme kernel memory.

This module does not fit the upstream response map or the reference-cell head.
Query bundles contain inputs only; their target outcomes are never accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frozen"))

from bioffn_v7.baselines import group_weights
from bioffn_v8.memory import kernel_geometry
from bioffn_v16.memory import MemoryModel, fit_memory, kernel_matrix
from bioffn_v16.biological_uncertainty import posterior_variance


SCHEMA = "biomeme-prepared-memory-v1"
IDENTITY = ("coordinate_system_id", "output_space_id", "baseline_id", "target_cell", "readout")
ROW_LABELS = ("ids", "groups", "names", "conditions")
COMMON = set(ROW_LABELS) | {"keys", "feature_names"}
FIELDS = {"training": COMMON | {"residuals", "calibration"},
          "update": COMMON | {"residuals", "calibration"},
          "query": COMMON | {"baseline", "dmso_center"}}


def guard_memory():
    if sys.platform == "darwin":
        p = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                           capture_output=True, text=True, check=False)
        if p.returncode == 0 and int(p.stdout.strip()) >= 4:
            raise RuntimeError("Critical memory pressure; numerical work stopped. Retry when memory is available.")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_bundle(arrays, metadata, kind):
    """Validate identities, dimensions and the declared input-only boundary."""
    if kind not in FIELDS:
        raise ValueError("Unknown bundle kind")
    allowed = FIELDS[kind]
    missing = allowed - {"dmso_center"} - set(arrays)
    extra = set(arrays) - allowed
    if missing or extra:
        raise ValueError(f"Invalid {kind} fields; missing={sorted(missing)}, unexpected={sorted(extra)}")
    if metadata.get("schema") != SCHEMA or metadata.get("kind") != kind:
        raise ValueError("Bundle schema or kind does not match the requested operation")
    for field in IDENTITY:
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Supply a nonempty {field}")
    if metadata.get("query_targets_used_in_preparation") is not False:
        raise ValueError("Declare query_targets_used_in_preparation=false")
    if kind in {"training", "update"}:
        if metadata.get("residual_definition") != "measured_target_minus_frozen_backbone":
            raise ValueError("Declare the residual definition")
        if metadata.get("calibration_residuals") != "whole_compound_out_of_fold":
            raise ValueError("Calibration residuals must use whole-compound out-of-fold backbone predictions")
    out = {key: np.asarray(value).copy() for key, value in arrays.items()}
    ids = out["ids"]
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("Supply at least one row")
    n = len(ids)
    for field in ROW_LABELS + ("feature_names",):
        a = out[field]
        if a.ndim != 1 or a.dtype.kind not in "US" or not all(str(v).strip() for v in a):
            raise ValueError(f"{field} must be a nonempty string vector")
        out[field] = a.astype(str)
    if any(len(out[k]) != n for k in ROW_LABELS):
        raise ValueError("Row labels are not aligned")
    if len(set(out["ids"])) != n:
        raise ValueError("Experiment IDs must be unique")
    if len(set(out["feature_names"])) != len(out["feature_names"]):
        raise ValueError("Output feature names must be unique")
    p = len(out["feature_names"])
    keys = np.asarray(out["keys"], dtype=float)
    if keys.ndim != 3 or keys.shape[1:] != (n, 39) or keys.shape[0] < 1 or not np.isfinite(keys).all():
        raise ValueError("keys must have finite shape (providers, rows, 39)")
    out["keys"] = keys
    field = "baseline" if kind == "query" else "residuals"
    values = np.asarray(out[field], dtype=float)
    if values.shape != (n, p) or p == 0 or not np.isfinite(values).all():
        raise ValueError(f"{field} must be finite and aligned with rows/output features")
    out[field] = values
    if kind == "query" and "dmso_center" in out:
        center = np.asarray(out["dmso_center"], dtype=float)
        if center.shape not in {(p,), (n, p)} or not np.isfinite(center).all():
            raise ValueError("dmso_center must be an output vector or one vector per query")
        out["dmso_center"] = center
    if kind != "query":
        mask = out["calibration"]
        if mask.dtype.kind != "b" or mask.shape != (n,):
            raise ValueError("calibration must be a boolean row mask")
        if kind == "training" and mask.sum() < 2:
            raise ValueError("At least two calibration rows are needed to freeze kernel geometry")
        if kind == "update" and mask.any():
            raise ValueError("Updates add reference measurements; the calibration/key preparation stays fixed")
        if set(out["groups"][mask]) & set(out["groups"][~mask]):
            raise ValueError("A calibration compound cannot also appear as a reference compound")
    return out, dict(metadata)


def load_bundle(prefix, kind):
    prefix = Path(prefix)
    meta = json.loads(prefix.with_suffix(".json").read_text())
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
        # Check names before decoding any arrays, including accidentally supplied outcomes.
        if set(archive.files) - FIELDS[kind]:
            raise ValueError(f"Unexpected fields in {kind} bundle: {sorted(set(archive.files) - FIELDS[kind])}")
        arrays = {key: archive[key] for key in archive.files}
    return validate_bundle(arrays, meta, kind)


def save_bundle(prefix, arrays, metadata, kind):
    arrays, metadata = validate_bundle(arrays, metadata, kind)
    prefix = Path(prefix)
    if prefix.with_suffix(".npz").exists() or prefix.with_suffix(".json").exists():
        raise FileExistsError(f"Bundle already exists: {prefix}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    write_json(prefix.with_suffix(".json"), metadata)


def _same_space(meta, other, arrays, candidate):
    for key in IDENTITY:
        if meta[key] != other[key]:
            raise ValueError(f"Mismatched {key}; reuse the fitted preparation and backbone")
    if not np.array_equal(arrays["feature_names"], candidate["feature_names"]):
        raise ValueError("Output feature names/order differ")
    if arrays["keys"].shape[0] != candidate["keys"].shape[0]:
        raise ValueError("Response-map provider count differs")


def _pack(value, blobs):
    if isinstance(value, dict):
        return {k: _pack(v, blobs) for k, v in value.items()}
    if isinstance(value, list):
        try:
            array = np.asarray(value)
        except ValueError:
            array = np.asarray([], dtype=object)
        if array.dtype.kind in "biufUS" and array.size > 32:
            name = f"fitted_blob_{len(blobs)}"
            blobs[name] = array
            return {"__array__": name}
        return [_pack(v, blobs) for v in value]
    return value


def _unpack(value, blobs):
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            return blobs[value["__array__"]].tolist()
        return {k: _unpack(v, blobs) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, blobs) for v in value]
    return value


@dataclass
class ResponseMemory:
    """Fitted frozen-parameter kernel memory with experiment-level identities."""
    arrays: dict
    metadata: dict
    models: list
    history: list

    def fingerprint(self):
        digest = hashlib.sha256(json.dumps({k:self.metadata[k] for k in IDENTITY}, sort_keys=True).encode())
        for key in sorted(self.arrays):
            value = np.ascontiguousarray(self.arrays[key])
            digest.update(json.dumps([key, value.dtype.str, value.shape]).encode())
            digest.update(value.tobytes())
        return digest.hexdigest()

    @classmethod
    def fit(cls, arrays, metadata, *, geometries=None, history=None):
        guard_memory()
        a, meta = validate_bundle(arrays, metadata, "training")
        source = np.flatnonzero(a["calibration"])
        refs = np.flatnonzero(~a["calibration"])
        order = np.r_[source, refs]
        a = {k: (v[:, order] if k == "keys" else v if k == "feature_names" else v[order])
             for k, v in a.items()}
        n = len(source)
        models = []
        if geometries is not None and len(geometries) != len(a["keys"]):
            raise ValueError("Geometry/provider count mismatch")
        for provider, z in enumerate(a["keys"]):
            guard_memory()
            geo = kernel_geometry(z[:n]) if geometries is None else geometries[provider]
            models.append(fit_memory(z[:n], a["residuals"][:n], a["groups"][:n], geometry=geo,
                reference_features=z[n:], reference_residual=a["residuals"][n:],
                reference_groups=a["groups"][n:], variant="main"))
        return cls(a, meta, models, list(history or []) + [dict(operation="fit", rows=len(order),
            calibration_rows=n, reference_rows=len(refs), query_targets_used=False)])

    def update(self, arrays, metadata):
        """Return a new memory, preserving the original model and coordinate system."""
        new, meta = validate_bundle(arrays, metadata, "update")
        _same_space(self.metadata, meta, self.arrays, new)
        if set(self.arrays["ids"]) & set(new["ids"]):
            raise ValueError("Update repeats an existing experiment ID")
        source_groups = set(self.arrays["groups"][self.arrays["calibration"]])
        if source_groups & set(new["groups"]):
            raise ValueError("Update cannot change calibration compounds")
        combined = {k: (v if k == "feature_names" else
                        np.concatenate((v, new[k]), axis=1 if k == "keys" else 0))
                    for k, v in self.arrays.items()}
        history = self.history + [dict(operation="add_measurements", added_ids=new["ids"].tolist())]
        return self.fit(combined, self.metadata, geometries=[m.geometry for m in self.models], history=history)

    def predict(self, arrays, metadata):
        guard_memory()
        query, meta = validate_bundle(arrays, metadata, "query")
        _same_space(self.metadata, meta, self.arrays, query)
        if set(query["ids"]) & set(self.arrays["ids"]):
            raise ValueError("Query experiment IDs overlap memory")
        if set(query["groups"]) & set(self.arrays["groups"]):
            raise ValueError("Query compounds overlap memory; remove newly measured compounds from candidates")
        n = len(self.arrays["ids"])
        g = group_weights(self.arrays["groups"])
        w = n * g
        means, weights, variances = [], [], []
        for model, zq in zip(self.models, query["keys"]):
            guard_memory()
            z = model.train_features
            gram = kernel_matrix(z, z, model.geometry)
            cross = kernel_matrix(z, zq, model.geometry)
            fitted = model.v15_model
            h, q = fitted.train_h, fitted.q
            hq = np.ones(len(zq)) / np.sqrt(q)
            system = gram * h[:, None] * h[None, :]
            system = (system + system.T) / 2
            system.flat[::n + 1] += 100. / w
            s = np.linalg.solve(system, (hq[:, None] * cross * h[None, :]).T).T
            complete = s + (1. - s.sum(1))[:, None] * (g / q)[None, :]
            value = model.predict(zq)
            np.testing.assert_allclose(complete.sum(1), 1., rtol=0, atol=1e-10)
            np.testing.assert_allclose(complete @ self.arrays["residuals"], value, rtol=1e-9, atol=1e-9)
            variance, _ = posterior_variance(gram, cross,
                np.full(len(zq), model.geometry["linear_scale"]), row_weights=w,
                train_h=h, query_h=hq, ridge=100.)
            means.append(value)
            weights.append(complete)
            variances.append(variance)
        correction = np.mean(means, axis=0)
        result = {k: query[k] for k in ROW_LABELS + ("feature_names",)}
        result.update(prediction=query["baseline"] + correction, baseline=query["baseline"],
            memory_correction=correction, weights=np.mean(weights, axis=0),
            latent_memory_variance=np.mean(variances, axis=0),
            memory_ids=self.arrays["ids"], memory_groups=self.arrays["groups"],
            memory_names=self.arrays["names"], memory_conditions=self.arrays["conditions"],
            memory_fingerprint=np.asarray(self.fingerprint()))
        if "dmso_center" in query:
            result["predicted_response_strength"] = np.mean((result["prediction"] - query["dmso_center"]) ** 2, axis=1)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        blobs = {}
        models = _pack([m.to_dict() for m in self.models], blobs)
        np.savez_compressed(directory / "memory.npz", **self.arrays, **blobs)
        write_json(directory / "memory.json", dict(schema=SCHEMA, metadata=self.metadata,
            models=models, history=self.history, arrays=list(self.arrays),
            sha256=sha256(directory / "memory.npz"), model="kernel-memory component",
            parameters=dict(lambda_value=100., bandwidth=1., variant="main")))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        record = json.loads((directory / "memory.json").read_text())
        if record.get("schema") != SCHEMA or sha256(directory / "memory.npz") != record.get("sha256"):
            raise ValueError("Memory format/checksum mismatch")
        with np.load(directory / "memory.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in record["arrays"]}
            models = [MemoryModel.from_dict(d) for d in _unpack(record["models"], archive)]
        arrays, meta = validate_bundle(arrays, record["metadata"], "training")
        if len(models) != len(arrays["keys"]):
            raise ValueError("Saved provider count mismatch")
        return cls(arrays, meta, models, record["history"])


def ranking_rows(result, score):
    """Descending ranks within declared condition groups, with stable ID ties."""
    field = {"uncertainty": "latent_memory_variance", "activity": "predicted_response_strength"}[score]
    if field not in result:
        raise ValueError("Activity ranking requires a user-supplied DMSO centre in the query bundle")
    rows = []
    for condition in sorted(set(result["conditions"])):
        ix = np.flatnonzero(result["conditions"] == condition)
        order = ix[np.lexsort((result["ids"][ix], -result[field][ix]))]
        for rank, i in enumerate(order, 1):
            rows.append(dict(condition=str(condition), rank=rank, query_id=str(result["ids"][i]),
                compound_id=str(result["groups"][i]), name=str(result["names"][i]),
                score_type=field, score=float(result[field][i])))
    return rows


def contribution_rows(result, memory, *, query_id=None, top=None):
    selected = np.arange(len(result["ids"]))
    if query_id is not None:
        selected = np.flatnonzero(result["ids"] == query_id)
        if len(selected) != 1:
            raise ValueError("Query ID is missing or ambiguous")
    if not np.array_equal(result["memory_ids"], memory.arrays["ids"]):
        raise ValueError("Prediction and memory identities differ")
    if str(result["memory_fingerprint"]) != memory.fingerprint():
        raise ValueError("Prediction belongs to different memory measurements or coordinates")
    rows = []
    for i in selected:
        weights = result["weights"][i]
        order = np.lexsort((memory.arrays["ids"], -np.abs(weights)))
        if top is not None:
            if top < 1:
                raise ValueError("top must be positive")
            order = order[:top]
        for rank, j in enumerate(order, 1):
            rows.append(dict(query_id=str(result["ids"][i]), query_name=str(result["names"][i]),
                rank_by_absolute_weight=rank, experiment_id=str(memory.arrays["ids"][j]),
                experiment_name=str(memory.arrays["names"][j]), compound_id=str(memory.arrays["groups"][j]),
                condition=str(memory.arrays["conditions"][j]), signed_weight=float(weights[j]),
                contribution_l2=float(np.linalg.norm(weights[j] * memory.arrays["residuals"][j]))))
    return rows


def write_csv(path, rows):
    path = Path(path)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(directory, result, memory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "prediction.npz", **result)
    write_csv(directory / "contributions.csv", contribution_rows(result, memory))
    write_csv(directory / "rank_uncertainty.csv", ranking_rows(result, "uncertainty"))
    if "predicted_response_strength" in result:
        write_csv(directory / "rank_activity.csv", ranking_rows(result, "activity"))
    error = float(np.max(np.abs(result["weights"] @ memory.arrays["residuals"] - result["memory_correction"])))
    write_json(directory / "prediction.json", dict(schema=SCHEMA, prediction_sha256=sha256(directory / "prediction.npz"),
        metadata=memory.metadata, query_targets_read=False, model="kernel-memory component",
        experiment_contribution_max_abs_reconstruction_error=error,
        uncertainty="latent KRR variance; excludes intercept, coordinate and reference-head uncertainty",
        activity="mean squared predicted distance from supplied DMSO centre; not a hit label or therapeutic effect",
        ranking="descending within condition; stable query-ID tie-break"))


def load_prediction(directory):
    directory = Path(directory)
    record = json.loads((directory / "prediction.json").read_text())
    if sha256(directory / "prediction.npz") != record["prediction_sha256"]:
        raise ValueError("Prediction checksum mismatch")
    with np.load(directory / "prediction.npz", allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}
