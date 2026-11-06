#!/usr/bin/env python3
"""Public LINCS example: build, predict, trace, update and rank a response memory."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
from pathlib import Path
import argparse
import json
import platform
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frozen"))
from biomeme.community import (SCHEMA, ResponseMemory, save_bundle, load_bundle, save_prediction,
                               write_json, guard_memory, sha256)
from biomeme.demo import dataset, subset, fit_G, predict_G, loadnp, codes


def prepare(output):
    """Read existing packaged fitting measurements and input-only query arrays."""
    guard_memory()
    output.mkdir(parents=True, exist_ok=False)
    data = ROOT / "data/u2os_demo"
    core, ref, query = (dataset(data, name) for name in ("core", "fold0_references", "fold0_query"))
    if hasattr(query, "target"):
        raise ValueError("Tutorial query inputs must not contain target outcomes")
    groups = sorted(set(ref.groups))
    before = np.flatnonzero(np.isin(ref.groups, groups[:5]))
    extra = np.flatnonzero(np.isin(ref.groups, groups[5:7]))
    qi = np.flatnonzero(np.isin(query.groups, sorted(set(query.groups))[:2]))
    model_g = fit_G(core)
    source = loadnp(data / "archived_source_residuals.npz")
    np.testing.assert_array_equal(source["ids"], core.ids)
    rr = ref.target.astype(float) - predict_G(model_g, ref)
    bank = loadnp(data / "coordinate_bank.npz")["rows"]
    indices = json.loads((data / "coordinate_indices.json").read_text())["0"]
    z = bank[indices["final_z"]].transpose(1, 0, 2)
    zq = bank[indices["query_z"]].transpose(1, 0, 2)
    del bank
    features = np.asarray(core.metadata["feature_names"], dtype=str)
    identity = dict(schema=SCHEMA, coordinate_system_id="LINCS-U2OS-ASG003-LKCP001-fold0-frozen-eight-response-providers",
        output_space_id="LINCS-1694-normalized-dmso-feature-intersection",
        baseline_id="LINCS-U2OS-ASG003-LKCP001-calibration64-multiwell-v16",
        target_cell="U2OS", readout="Cell Painting morphology",
        query_targets_used_in_preparation=False,
        provenance="Packaged public LINCS cpg0004-lincs processed fitting data; see DATA_SOURCES.md",
        residual_definition="measured_target_minus_frozen_backbone",
        calibration_residuals="whole_compound_out_of_fold")

    def labels(d):
        names = np.asarray([f"{r['compound_id']} / {r['cell']} / {r['layout']} / {r['target_time']}h / step{r['task']}"
                            for r in d.metadata["row_inventory"]])
        return dict(ids=d.ids, groups=d.groups, names=names, conditions=codes(d), feature_names=features)

    initial_ref = subset(ref, before)
    initial = {k: (labels(core)[k] if k == "feature_names" else np.r_[labels(core)[k], labels(initial_ref)[k]])
               for k in labels(core)}
    initial.update(keys=np.concatenate((z[:, :len(core.ids)], z[:, len(core.ids) + before]), axis=1),
        residuals=np.r_[source["residuals"], rr[before]],
        calibration=np.r_[np.ones(len(core.ids), dtype=bool), np.zeros(len(before), dtype=bool)])
    update = labels(subset(ref, extra))
    update.update(keys=z[:, len(core.ids) + extra], residuals=rr[extra], calibration=np.zeros(len(extra), dtype=bool))
    qdata = subset(query, qi)
    request = labels(qdata)
    request.update(keys=zq[:, qi], baseline=predict_G(model_g, qdata),
                   dmso_center=np.zeros(len(features)))
    save_bundle(output / "training", initial, dict(identity, kind="training"), "training")
    save_bundle(output / "new_measurements", update, dict(identity, kind="update"), "update")
    save_bundle(output / "queries", request, dict(identity, kind="query"), "query")
    write_json(output / "PREPARATION.json", dict(
        training_compounds=len(set(initial["groups"])), calibration_compounds=len(set(core.groups)),
        initial_reference_compounds=5, added_reference_compounds=2, query_compounds=2,
        providers=z.shape[0], output_features=len(features), query_outcomes_read=False,
        calibration_residuals="Existing cross-fitted residuals, matched by full experiment IDs",
        dmso_center="Zero vector in the packaged same-plate DMSO-centred output space",
        query_file_fields=sorted(request),
        source_files={name: sha256(data / name) for name in
                      ("core.npz", "fold0_references.npz", "fold0_query.npz", "coordinate_bank.npz",
                       "coordinate_indices.json", "archived_source_residuals.npz")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New tutorial directory")
    parser.add_argument("--prepare-only", action="store_true", help="Export inputs for the stepwise CLI tutorial")
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    prepare(args.output / "inputs")
    if args.prepare_only:
        print(f"Prepared tutorial inputs at {args.output / 'inputs'}")
        return
    training = load_bundle(args.output / "inputs/training", "training")
    query = load_bundle(args.output / "inputs/queries", "query")
    added = load_bundle(args.output / "inputs/new_measurements", "update")
    memory = ResponseMemory.fit(*training)
    memory.save(args.output / "memory_before")
    before = memory.predict(*query)
    save_prediction(args.output / "predictions_before", before, memory)
    updated = memory.update(*added)
    updated.save(args.output / "memory_after")
    # Exercise the saved artifact as well as the in-memory update.
    after = ResponseMemory.load(args.output / "memory_after").predict(*query)
    save_prediction(args.output / "predictions_after", after, updated)
    direct = ResponseMemory.fit(updated.arrays, updated.metadata).predict(*query)
    error = float(np.max(np.abs(after["prediction"] - direct["prediction"])))
    np.testing.assert_allclose(after["prediction"], direct["prediction"], rtol=1e-10, atol=1e-10)
    before_again = memory.predict(*query)
    np.testing.assert_array_equal(before_again["prediction"], before["prediction"])
    record = dict(status="passed", scope="Software tutorial, not a new scientific evaluation",
        model="BioMeme kernel-memory component; reference-cell transfer is not fitted in this tutorial",
        elapsed_seconds=time.perf_counter() - start, platform=platform.platform(), python=sys.version,
        numpy=np.__version__, query_outcomes_read=False,
        update_matches_fresh_fit_max_abs_error=error, original_memory_unchanged=True,
        training_rows_before=len(memory.arrays["ids"]), training_rows_after=len(updated.arrays["ids"]),
        query_rows=len(query[0]["ids"]), feature_count=len(memory.arrays["feature_names"]),
        before_reconstruction_max_abs_error=float(np.max(np.abs(before["weights"] @ memory.arrays["residuals"] - before["memory_correction"]))),
        after_reconstruction_max_abs_error=float(np.max(np.abs(after["weights"] @ updated.arrays["residuals"] - after["memory_correction"]))),
        prediction_change_l2=float(np.linalg.norm(after["prediction"] - before["prediction"])))
    write_json(args.output / "validation.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
