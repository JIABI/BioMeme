# Build a response memory from your measurements

This workflow exposes BioMeme's **kernel-memory component**: build memory → predict the next dose → identify the supporting experiments → add measurements → rank the next candidates. It calls the same frozen memory fitter, kernel and uncertainty functions used by the manuscript reproduction. The RBF bandwidth is 1 and the ridge penalty is 100; this interface does not select a new model or change the estimator.

The entry boundary is **prepared morphology data**: fixed 39-coordinate keys, measured response residuals and backbone predictions. The keys contain 16 source-morphology coordinates, 16 RNA-derived response coordinates and seven dose/time coordinates. The manuscript uses eight fixed response-map providers. This API supports one or more already prepared providers and averages their predictions. The response map, coordinate preparation and backbone must be fitted using allowed training/calibration data and then applied unchanged to new experiments. This is an interface to the memory component; the full reference-cell transfer-head fit and its selection remain in `run_demo.py` and the complete research implementation.

## Run the public LINCS tutorial

Install the dependencies and fetch the example inputs as described in the main README, then run with the installed environment:

```bash
.venv/bin/python tutorial_memory.py --output results/memory_tutorial
```

The tutorial uses the existing public LINCS U2OS morphology arrays in `data/u2os_demo/`. It builds a memory from 64 calibration compounds and five reference compounds, predicts eight next-dose conditions for two separate query compounds, adds two further reference compounds and predicts the same queries again. All 1,694 output features and eight key providers are retained. The additional measurements come from the packaged fitting references; **no query outcomes are opened**. The ordering is a reproducible software example, not a screening-performance analysis.

The tutorial writes:

- `inputs/`: reusable training, query and new-measurement bundles plus preparation provenance.
- `memory_before/` and `memory_after/`: fitted memory artifacts, checksums and update history.
- `predictions_before/` and `predictions_after/`: predicted profiles, full signed experiment weights, a named contribution table, and uncertainty/activity rankings.
- `validation.json`: verifies that adding measurements matches fitting the combined memory from scratch, that the original memory stays unchanged, and that named experiment contributions reconstruct the prediction.

The tutorial uses a zero DMSO centre because its output features already use the packaged same-plate DMSO-centred normalization. For another assay, provide its appropriate centre explicitly.

To run the steps separately, activate the environment (`source .venv/bin/activate`) and choose a fresh directory:

```bash
python tutorial_memory.py --output results/memory_steps --prepare-only
python run_memory.py fit --data results/memory_steps/inputs/training --output results/memory_steps/memory_before
python run_memory.py predict --memory results/memory_steps/memory_before --data results/memory_steps/inputs/queries --output results/memory_steps/predictions_before
python run_memory.py update --memory results/memory_steps/memory_before --data results/memory_steps/inputs/new_measurements --output results/memory_steps/memory_after
python run_memory.py predict --memory results/memory_steps/memory_after --data results/memory_steps/inputs/queries --output results/memory_steps/predictions_after
python run_memory.py rank --prediction results/memory_steps/predictions_after --by uncertainty --output results/memory_steps/next_uncertain.csv
python run_memory.py rank --prediction results/memory_steps/predictions_after --by activity --output results/memory_steps/next_active.csv
```

`predictions_after/contributions.csv` lists every named experiment for every query. To export only the largest ten absolute weights for a specific query, copy its complete ID from `rank_uncertainty.csv`:

```bash
python run_memory.py trace --memory results/memory_steps/memory_after --prediction results/memory_steps/predictions_after --query-id 'YOUR_COMPLETE_QUERY_ID' --top 10 --output results/memory_steps/query_trace.csv
```

Output directories and CSV files must be new. Existing runs are preserved.

## Data contract

Each bundle consists of a NumPy archive and a JSON metadata file sharing the same prefix, for example `training.npz` and `training.json`. Pass `--data training`, without an extension. Object arrays and pickle are not used.

| Array | Shape and contents | Used in |
|---|---|---|
| `ids` | `(n,)` unique string ID for each experiment/condition | All bundles |
| `groups` | `(n,)` stable compound identity; the same chemical always has the same identity | All bundles |
| `names` | `(n,)` readable experiment labels, such as compound, cell, time and dose step | All bundles |
| `conditions` | `(n,)` condition-group labels; ranks are calculated within each label | All bundles |
| `feature_names` | `(p,)` unique output feature names in a fixed order | All bundles |
| `keys` | `(providers, n, 39)` finite, previously prepared coordinates | All bundles |
| `residuals` | `(n, p)` measured target minus the appropriate frozen-backbone prediction | Training and update |
| `calibration` | `(n,)` boolean mask; selects the calibration rows used to fix kernel geometry | Training and update |
| `baseline` | `(n, p)` backbone predictions for the next-dose query output | Query only |
| `dmso_center` | Optional `(p,)` or `(n, p)` DMSO centre in the same output space | Query only |

Calibration residuals must come from **whole-compound out-of-fold** backbone predictions. Reference residuals use the backbone fitted on the calibration panel. Key standardization, PCA, block scaling and RNA-response maps must not use query targets. Raw RNA is not automatically transformed by this entry; supply keys produced by a frozen, training-only preparation pipeline. The public tutorial demonstrates the exact exported coordinate boundary. `SOURCE_FUNCTIONS.json`, `DATA_SOURCES.md` and the full-study source archive in the [Zenodo software ZIP](https://doi.org/10.5281/zenodo.24008759) document upstream preparation.

The following metadata is required in each JSON file:

```json
{
  "schema": "biomeme-prepared-memory-v1",
  "kind": "training",
  "coordinate_system_id": "my-screen-frozen-key-v1",
  "output_space_id": "my-screen-normalized-features-v1",
  "baseline_id": "my-screen-calibration-backbone-v1",
  "target_cell": "my-cell-line",
  "readout": "Cell Painting morphology",
  "query_targets_used_in_preparation": false,
  "residual_definition": "measured_target_minus_frozen_backbone",
  "calibration_residuals": "whole_compound_out_of_fold"
}
```

Set `kind` to `training`, `query` or `update`. The final two residual declarations are required for training and update bundles. The coordinate-system, output-space and backbone identifiers must match across all bundles. Feature order and provider count are checked. These identifiers document the caller's preparation; the API cannot establish how externally supplied arrays were generated.

Queries must be compound-disjoint from the current memory. Query files accept only the listed input arrays: an accidental `target`, `residuals` or other outcome field is rejected before any array is decoded. When a previously predicted compound is measured and added, remove it from the query candidate set. An update adds new reference rows with `calibration=false`, cannot repeat experiment IDs or change calibration compounds, and retains the fitted key geometry. Updating the full BioMeme predictor additionally requires refitting/selecting its reference-cell transfer head, as in the original workflow.

## Python API

From the package directory:

```python
from biomeme.community import ResponseMemory, load_bundle, save_prediction

memory = ResponseMemory.fit(*load_bundle("training", "training"))
memory.save("memory_v1")
prediction = memory.predict(*load_bundle("queries", "query"))
save_prediction("predictions_v1", prediction, memory)

updated = memory.update(*load_bundle("new_measurements", "update"))
updated.save("memory_v2")
```

For array-based use, pass `(arrays, metadata)` directly. `save_bundle(prefix, arrays, metadata, kind)` validates and writes this format. The original `memory` object remains usable after `update`.

## Interpreting the outputs

The memory prediction is exactly `baseline + weights @ measured_residuals`. The complete experiment weights sum to one and can be negative. `contributions.csv` reports the signed weight and the norm of its response-vector contribution; its displayed order is by absolute weight. This is a decomposition of the predictor, not an estimate of causal biological effects. The full matrix and experiment identities remain in `prediction.npz`.

Uncertainty ranks the **latent memory KRR variance**, excluding uncertainty in the fixed residual intercept, coordinates and reference-cell head. Activity ranking is available only with an explicit DMSO centre and sorts the mean squared distance of the prediction from that centre. It describes predicted morphological response strength; it does not assign a hit threshold or a therapeutic-effect label. Rankings are descending within the supplied condition groups, with query IDs breaking ties. No top-K selection, statistical test or evaluation target is used by these commands.

The interface uses the same pinned dependencies as the original package. Its synthetic tests can be run separately with `python -m pytest -q tests/test_community.py`. The interface and tutorial were validated on macOS in the archived release; platform history and the scope of current checks are documented in [Reproducibility](docs/REPRODUCIBILITY.md).
