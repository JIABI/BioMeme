# BioMeme: biological response memory for virtual cells

BioMeme predicts next-dose cellular responses from measured lower-dose states. It combines a kernel memory of responses in the target cell with the same compound’s response in a reference cell line. Predictions can be decomposed into named experimental contributions, and new measurements can be added to the memory through a linear solve.

This repository provides:

- **A complete fit–predict–score example** for one U2OS Cell Painting task, including calibration, memory and the reference-cell transfer head.
- **A response-memory interface** for fitting, prediction, experiment-level contributions, updates and candidate rankings from prepared measurements.
- **Statistical reproduction** of the original common loss comparisons and the mechanism-weight analysis from archived inputs.

The [published Zenodo collection](https://doi.org/10.5281/zenodo.24008759) contains the full study’s data, code, predictions, supplementary data and provenance. This GitHub package is version **4.22**; its scientific implementation is unchanged from software version **4.21**.

## Quick start

Requirements: Python **3.12**, a CPU, about **3 GB of free RAM**, and roughly **2 GB of free disk space** for the environment, download and outputs. The first run downloads the 382 MB software archive from Zenodo and extracts 52 input files (135 MB); it does not download the full 20.79 GB study collection. A working HTTPS connection is needed for dependencies and the initial data download.

From the repository directory:

```bash
BIOMEME_PYTHON=python3.12 bash run_all.sh
```

The command creates `.venv`, installs the pinned dependencies, downloads and verifies the example inputs, fits four held-out folds, replays the archived statistics, and runs the tests. It writes results to `results/`. The numerical entry points use one CPU thread. No GPU is needed.

Use a new output directory for another run:

```bash
BIOMEME_PYTHON=python3.12 bash run_all.sh results/second_run
```

If the published software ZIP is already on your computer, reuse it:

```bash
BIOMEME_DATA_ARCHIVE=/path/to/BioMeme_4_22_software.zip BIOMEME_PYTHON=python3.12 bash run_all.sh
```

The archive and every extracted input are checked against fixed SHA-256 hashes. Existing results are preserved. Check already installed inputs without downloading with `.venv/bin/python scripts/fetch_data.py --check`. Downloaded data, local environments and generated results are excluded from Git.

## What the example reproduces

The fitting example covers the complete U2OS **ASG003 → LKCP001** direction: **101 query compounds, 404 conditions, 1,694 morphology features and four outer folds**. It refits the calibration backbone, residual memory and selected transfer head from real processed arrays. Predictions are saved and sealed before held-out targets are opened for scoring.

Expected compound-averaged results for this direction are:

| Model | MSE | Centred direction error |
|---|---:|---:|
| Calibration backbone | 5.5279922424 | 0.6196394236 |
| Kernel memory | 4.7936075542 | 0.4390701662 |
| BioMeme | 3.6207619686 | 0.3315620584 |

These are the example’s results. The paper’s pooled U2OS evaluation uses 292 compounds across three directions.

`reproduce_stats.py` recomputes **30 original MSE/direction comparisons**, including the **nine preregistered direction comparisons**, across U2OS, sci-Plex3 and Tahoe. It also replays the A5 mechanism-weight analysis. This step starts from archived losses, fitted weights and bootstrap draws; it does not refit every baseline in the paper.

Important outputs:

- `results/demo/validation.json`: population, prediction checks, scores and timings.
- `results/demo/fold*/prediction.npz`: predictions and the four additive model terms.
- `results/demo/fold*/memory_weights.npz`: signed experiment weights and measured residuals.
- `results/demo/fold*/prediction_seal.json`: saved prediction identity before scoring.
- `results/statistics/core/validation.json`: agreement with the archived statistical comparisons.
- `results/statistics/a5/REPLAY_VERIFICATION.json`: mechanism-weight replay checks.

The current package passed a fresh-environment macOS run: all four folds matched the archived predictions, all 30 statistical comparisons were reproduced, and all 28 tests passed. The real-data memory-update tutorial also passed. See the [release validation record](provenance/RELEASE_VALIDATION.json) and [Reproducibility](docs/REPRODUCIBILITY.md) for details and platform scope.

## Use your own measurements

[USE_YOUR_DATA.md](USE_YOUR_DATA.md) describes the prepared-data contract and the Python and command-line interfaces. This interface exposes the **kernel-memory component** with fixed 39-coordinate morphology/RNA/dose/time keys. It expects prepared coordinates, measured residuals and backbone predictions. The full reference-cell transfer-head fit is demonstrated in `run_demo.py`.

After the quick start, run the real-data tutorial:

```bash
.venv/bin/python tutorial_memory.py --output results/memory_tutorial
```

It builds a memory, predicts separate queries, adds measurements, verifies the update and exports experiment contributions and rankings. Updating the memory preserves its coordinate system. Updating the full BioMeme predictor also requires refitting/selecting the transfer head.

For a small interface check without downloading study data:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-test.txt
.venv/bin/python -m pytest -q tests/test_community.py
```

## Run the steps separately

After installing the dependencies:

```bash
.venv/bin/python scripts/fetch_data.py
.venv/bin/python run_demo.py --output results/demo
.venv/bin/python reproduce_stats.py --output results/statistics
BIOMEME_DEMO_RESULTS=results/demo BIOMEME_STATS_RESULTS=results/statistics   .venv/bin/python -m pytest -q tests
```

The repository uses scripts directly; it does not require a system-wide package installation. `environment.yml` provides an alternative conda environment description with the same dependency pins.

## Repository layout

| Path | Purpose |
|---|---|
| `biomeme/` | Demo, prepared-data interface and numerical validation |
| `frozen/` | Archived scientific model functions |
| `run_demo.py` | Complete four-fold fit–predict–score example |
| `run_memory.py`, `tutorial_memory.py` | Prepared-data memory workflow and tutorial |
| `reproduce_stats.py`, `statistics/` | Statistical replay code; inputs fetched from Zenodo |
| `scripts/` | Verified data retrieval |
| `tests/` | Reproduction, memory-interface and retrieval tests |
| `provenance/` | Source correspondence and release records |

The original full-study source, historical validation records and preparation inventories remain in the Zenodo software ZIP under `BioMeme_4_22/software/BioMeme/`. That archive is the reference for the wider study. The executable workflow above is the portable entry point.

## Licence and citation

Code is licensed under the [MIT Licence](LICENSE). Data, microscopy and annotations retain their upstream licences and attribution requirements; see [DATA_SOURCES.md](DATA_SOURCES.md). The code licence does not relicense the mixed-origin study archive.

Please cite the versioned collection:

> Bi, J. (2026). *BioMeme: biological response memory for virtual cells — data, code and reproducibility materials* (Version 4.22). Zenodo. https://doi.org/10.5281/zenodo.24008759

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff). [CHANGELOG.md](CHANGELOG.md) records the GitHub packaging changes.
