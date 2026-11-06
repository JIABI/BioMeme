# Reproducibility and scope

## Two reproduction levels

The complete fitting example in `run_demo.py` refits one archived U2OS direction, ASG003 → LKCP001. It includes 101 held-out query compounds and four outer folds. The calibration set contains 64 compounds; each fold uses 75 or 76 allowed target-cell reference compounds. The example preserves the original compound identities, row order, model grids, seeds and score definitions.

It refits condition-specific calibration, nested calibration residuals, eight residual-memory learners, their own out-of-fold errors, and the gated transfer-head grid. No fitted memory or transfer-head coefficients are supplied as inputs. Newly fitted predictions are written and hashed before the corresponding scoring-only target is opened.

The separate statistics replay recomputes 30 common loss comparisons across U2OS (292 compounds), sci-Plex3 (120) and Tahoe (277), retaining the archived 10,000 bootstrap draws and within-family Holm corrections. It also replays A5 from stored signed weights and typed annotations. Replaying statistics and refitting all models are different operations; the full-study source and remaining analysis packages are in the [Zenodo collection](https://doi.org/10.5281/zenodo.24008759).

## Prepared-input boundary

The example starts from real processed morphology arrays, measured A549 donor responses, and fixed 39-coordinate keys: 16 morphology coordinates, 16 RNA-response coordinates and seven condition coordinates. The eight RNA-response maps were fitted on independent Pilot 1 data with no-intercept ridge regression and a penalty of 10,000. Upstream maps and coordinate preparation are not retrained by this example.

`DATA_SOURCES.md` documents the data origins. `SOURCE_FUNCTIONS.json` maps scientific functions to the historical implementation. Its historical archive aliases describe provenance; they are not required local paths.

The prepared-data memory interface uses this same coordinate boundary. Its update operation adds measurements to memory while retaining the fitted geometry. Updating the complete predictor also requires refitting/selecting the reference-cell transfer head.

## Verification

The full command checks the four fitted folds against archived predictions with relative error tolerance `1e-8`. It verifies exact additive reconstruction, signed memory-weight reconstruction, query separation and the original direction-test decisions. Recomputed statistics must agree with their archived values within the declared numerical tolerances.

Refitted source residuals are compared over the complete array using both relative L2 error and maximum absolute difference, each at most `1e-10`. File hashes identify the immutable inputs; floating-point numerical acceptance does not require byte-identical arrays across platforms.

The synthetic interface tests cover fit/predict/update consistency, named contributions, rejection of outcome fields in query inputs, incompatible coordinate spaces, repeated measurements, within-condition ranking and command-line use outside the repository directory. The GitHub Actions workflow runs these small checks and data-retrieval tests on Python 3.12. Its execution status is available only after the repository is uploaded and the workflow has run; the workflow does not run the full study.

## Platform history

The archived workflow was fitted and checked on macOS ARM64 with Python 3.12.14. An earlier Linux x86_64 run with Python 3.12.3 reproduced predictions within `3.2e-15` and matched model selection and statistics, but failed a byte-level source-residual assertion. Version 4.9.1 replaced that assertion with the numerical checks described above. The corrected full workflow and prepared-data interface were subsequently tested on macOS. This history does not constitute a completed Linux run of the current GitHub package.

Historical receipts are preserved in the published software ZIP under `BioMeme_4_22/software/BioMeme/validation/`. Current package checks are recorded separately under `provenance/`; they should not be inferred from older receipts.

## Data integrity and provenance

`scripts/fetch_data.py` retrieves the fixed software archive for Zenodo record `24008759`. It checks the archive SHA-256 before extracting the allowlisted example and statistics inputs, then checks their individual hashes. An existing archive can be supplied with `--archive`; `run_all.sh` accepts the same path through `BIOMEME_DATA_ARCHIVE`.

The repository's scientific code comes from the published 4.21 implementation. `provenance/SCIENTIFIC_SOURCE_MAP.json` records the upstream archive members and hashes. GitHub-specific documentation and retrieval wrappers are packaging additions. The data collection and GitHub package are labelled 4.22; this version label does not imply a new fitted model or changed scientific result.

The original source archive includes historical filenames and protocol records. Those records remain intact on Zenodo to preserve the registered evidence trail. All runnable entry points in this repository use script-relative data paths or explicit arguments.
