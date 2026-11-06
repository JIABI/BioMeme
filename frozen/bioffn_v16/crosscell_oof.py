"""Full-output OOF residual errors for the scalar cross-cell correction.

Inputs are already prepared eight-fold source/reference packs. No raw data,
outer-query outcomes or donor measurements are loaded here. Neural models use
the externally selected configuration and one fixed calibration seed; this is
selected-configuration OOF, not fully nested hyperparameter selection.
"""
from __future__ import annotations

import numpy as np

from bioffn_v8.memory import kernel_geometry
from bioffn_v13.linear import fit_path
from biomeme.optional import neural
from bioffn_v14.memory import fit_memory

from .crosscell import reconstruct_oof_errors, validate_oof_coverage


METHODS = ("G", "memory", "linear", "mlp_gelu64", "mlp_gelu256")
CV_SEED = 20261008
CV_FOLDS = 8
LEARNING_RATE = .001
WEIGHT_DECAYS = (3., 10., 30., 100.)


def _arrays(pack, *, expected_input_coordinates=39):
    z, zv = np.asarray(pack["z"], float), np.asarray(pack["z_valid"], float)
    residual, valid = np.asarray(pack["R"], float), np.asarray(pack["R_valid"], float)
    if (z.ndim != 3 or min(z.shape) < 1 or z.shape[2] != expected_input_coordinates
            or zv.ndim != 3 or min(zv.shape) < 1
            or zv.shape[0] != z.shape[0] or zv.shape[2] != z.shape[2]):
        raise ValueError(f"OOF packs need aligned full {expected_input_coordinates}-coordinate provider features")
    if (residual.ndim != 2 or residual.shape[0] != z.shape[1] or residual.shape[1] < 1
            or valid.shape != (zv.shape[1], residual.shape[1])
            or len(pack["groups"]) != len(residual)
            or len(pack["ids"]) != len(residual)
            or len(pack["ids_valid"]) != len(valid)):
        raise ValueError("OOF residuals, IDs and full-output feature dimensions must align")
    if not all(np.isfinite(array).all() for array in (z, zv, residual, valid)):
        raise ValueError("OOF feature and residual arrays must be finite")
    source_rows = len(pack["audit"]["source_train"])
    if source_rows < 1 or source_rows > z.shape[1]:
        raise ValueError("OOF pack must identify its leading source-training rows")
    if source_rows + len(pack["audit"]["reference_train"]) != len(residual):
        raise ValueError("OOF source/reference training row inventory differs")
    return z, zv, residual, valid, source_rows


def generate_oof_errors(packs, method, *, ids, groups, query_groups,
                        selected_epoch=None, selected_decay=None, callback=None,
                        expected_input_coordinates=39):
    """Return errors in the requested ``ids``/``groups`` order, exactly once.

    ``errors = R_valid - residual_prediction`` already equals the error of
    the complete predictor because each supplied ``R_valid`` subtracts its
    fold-local G. Neural parameters are fixed externally, never reselected.
    The original 39-coordinate inventory remains the default. K3 neural
    inputs may explicitly request 71 coordinates; fixed methods remain at 39.
    The returned plain dictionary contains arrays, fold audits and model
    audits, and can be serialized with the existing pickle/JSON-safe tools.
    """
    packs = list(packs)
    if method not in METHODS:
        raise ValueError(f"Unknown OOF method: {method}")
    if len(packs) != CV_FOLDS or {pack["fold"] for pack in packs} != set(range(CV_FOLDS)):
        raise ValueError("Exactly eight distinct original OOF folds are required")
    is_neural = method.startswith("mlp_")
    if (isinstance(expected_input_coordinates, (bool, np.bool_))
            or not isinstance(expected_input_coordinates, (int, np.integer))
            or expected_input_coordinates not in (39, 71)):
        raise ValueError("Expected input coordinates must be the integer 39 or 71")
    if expected_input_coordinates == 71 and not is_neural:
        raise ValueError("71-coordinate OOF inputs are permitted only for neural methods")
    if is_neural:
        if (isinstance(selected_epoch, (bool, np.bool_))
                or not isinstance(selected_epoch, (int, np.integer))
                or not 0 <= selected_epoch <= 400 or selected_decay not in WEIGHT_DECAYS):
            raise ValueError("MLP requires the externally selected epoch 0..400 and decay from the frozen grid")
    # Check every fold before the first fit, including late-fold query overlap.
    validate_oof_coverage(packs, ids=ids, groups=groups, query_groups=query_groups)
    arrays = [_arrays(pack, expected_input_coordinates=expected_input_coordinates) for pack in packs]
    if len({(z.shape[0], z.shape[2], residual.shape[1])
            for z, _, residual, _, _ in arrays}) != 1:
        raise ValueError("All OOF folds must preserve the provider/input/output dimensions")
    predictions, model_audits = [], []
    for pack, (z, zv, residual, valid, source_rows) in zip(packs, arrays):
        fold = int(pack["fold"])
        if callback is not None:
            callback(dict(stage="crosscell_oof_fold_start", method=method, fold=fold,
                          folds=CV_FOLDS, training_rows=len(residual), validation_rows=len(valid)))
        if method == "G":
            prediction = np.zeros_like(valid)
            audits = [dict(method="G", residual_prediction="zero",
                           base_prediction="already subtracted in supplied R_valid",
                           full_output_dim=valid.shape[1])]
        elif is_neural:
            def update(record, current_fold=fold):
                if callback is not None:
                    callback(dict(stage="crosscell_oof_neural_training", method=method,
                                  fold=current_fold, folds=CV_FOLDS, detail=record))
            fitted = neural.fit_predict(z, residual, pack["groups"], {"valid": zv},
                width=64 if method == "mlp_gelu64" else 256, activation="gelu",
                seed=CV_SEED, weight_decay=float(selected_decay), epochs=int(selected_epoch),
                lr=LEARNING_RATE, callback=update)
            prediction = fitted["predictions"]["valid"]
            audits = [fitted["audit"]]
            del fitted
        else:
            prediction = np.zeros_like(valid)
            audits = []
            for provider in range(z.shape[0]):
                if method == "memory":
                    geometry = kernel_geometry(z[provider, :source_rows])
                    model = fit_memory(z[provider], residual, pack["groups"], np.ones(len(residual)),
                        geometry=geometry, kind="and", lambda_value=100., bandwidth=1.)
                    value = model.predict(zv[provider], np.ones(len(valid)))
                else:
                    model = fit_path(z[provider], residual, pack["groups"], [1000.])[1000.]
                    value = model.predict(zv[provider])
                prediction += value / z.shape[0]
                audits.append(dict(model.audit, provider=provider,
                                   geometry_source_rows=source_rows if method == "memory" else None))
        if prediction.shape != valid.shape or not np.isfinite(prediction).all():
            raise ValueError("OOF model must predict every original output coordinate")
        predictions.append(prediction)
        model_audits.append(dict(fold=fold, models=audits))
        if callback is not None:
            callback(dict(stage="crosscell_oof_fold_complete", method=method, fold=fold,
                          folds=CV_FOLDS, full_output_dim=valid.shape[1]))
    result = reconstruct_oof_errors(packs, predictions, ids=ids, groups=groups,
                                    query_groups=query_groups)
    result["ids_valid"] = result["ids"]
    result["groups_valid"] = result["groups"]
    result["method"] = method
    result["model_audits"] = model_audits
    result["audit"].update(method=method, fold_count=CV_FOLDS,
        output_dim=arrays[0][2].shape[1], output_compression=False,
        input_coordinates=arrays[0][0].shape[2],
        expected_input_coordinates=int(expected_input_coordinates),
        provider_members=arrays[0][0].shape[0],
        provider_prediction_aggregation="arithmetic mean of full-output predictions",
        residual_error="R_valid minus fold residual predictor; fold G already removed",
        outer_query_outcomes_used=False,
        hyperparameter_selection_repeated_in_fold=False,
        oof_scope=("selected-configuration OOF; not fully nested hyperparameter selection"
                   if is_neural else "fixed-configuration chemical OOF"),
        calibration_seed=CV_SEED if is_neural else None,
        calibration_seeds_per_family=1 if is_neural else None,
        final_seed_policy="one family coefficient shared by all five final seeds" if is_neural else None,
        selected_epoch=int(selected_epoch) if is_neural else None,
        selected_decay=float(selected_decay) if is_neural else None,
        learning_rate=LEARNING_RATE if is_neural else None)
    return result
