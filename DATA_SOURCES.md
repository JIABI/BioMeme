# Data sources and processing boundary

The example is the complete archived U2OS `ASG003 → LKCP001` direction: 101 query compounds, 404 query conditions, 1,694 morphology features, four disjoint outer query folds, and eight fixed key providers. The source FIT calibration inventory contains 64 compounds (256 conditions). Each outer fold uses all 75 or 76 permitted reference compounds; 26 or 25 compounds are withheld for scoring. Exact row identities and source inventories accompany the arrays.

Morphology profiles originate from the public LINCS Cell Painting data, batch `2017_12_05_Batch2`, with the manuscript's fixed plate, profile and feature processing. The upstream processed profiles are CC0; the LINCS repository code is BSD-3-Clause. Sources:

- LINCS Cell Painting profiles: https://github.com/broadinstitute/lincs-cell-painting/blob/master/profiles/README.md
- LINCS data/code terms: https://github.com/broadinstitute/lincs-cell-painting
- Data record: https://doi.org/10.5281/zenodo.5008187
- Cell Painting Gallery data terms: https://github.com/broadinstitute/cellpainting-gallery/blob/main/LICENSE

The underlying public morphology profiles are covered by CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/); this package preserves their source acknowledgement. Original source code in this package uses the MIT licence in `LICENSE`; this software licence does not replace upstream data or third-party software terms. BRD compound identities and public mechanism annotations are preserved for attribution and reproducibility. Annotation provenance for A5 is documented in `statistics/a5/README.md` and its original input files.

`core.npz` and the reference files hold the real processed input/target morphology arrays. Query input files contain no target array. Scoring-only files hold withheld outcomes, frozen scoring centres and archived expected predictions; the demo opens the appropriate scoring file after saving and hashing its newly fitted predictions.

The 39-coordinate keys are real frozen processed arrays: 16 morphology coordinates, 16 predicted RNA-response coordinates and seven condition coordinates. The external RNA-response map consists of eight no-intercept ridge providers with penalty 10,000 fitted on independent Pilot1 data. Their inference and fold-local source-only coordinate transformations were performed upstream and are exported here. No neural provider is fitted or required. This package starts at this processed-data boundary. It refits the calibration backbone, nested calibration residuals, all residual-memory coefficients, memory's own out-of-fold errors, and the complete gated transfer-head selection. Fitted memory or head coefficients are not inputs. The A549 donor arrays are the matched measured reference-cell responses and their four-condition trajectories.

`SOURCE_FUNCTIONS.json` identifies every original model source and copied function. Numerical model functions are copied from the archived confirmation snapshot. The optional neural import is redirected to an explicit unavailable-component adapter; the memory execution path is unchanged. The published software ZIP preserves `BioMeme_4_22/software/BioMeme/export_records/`, which records source artefacts read during export. `data/u2os_demo/COMPACTION.json` records lossless row banking: reconstructed coordinate arrays and donor values were checked byte-for-byte before duplicated export files were removed.

The statistics entry point uses the original 292-compound U2OS, 120-compound sci-Plex3 and 277-compound Tahoe populations. These are different from the 101-compound fitting demo. Archived per-compound losses preserve the original order of seed-score, condition and target averaging. Exact original 10,000-draw matrices and formal expected results accompany them; `statistics/core/PROVENANCE.json` gives their source files. Tahoe uses its accepted tuned-kernel main analysis and the original matched-well scoring target. A5 uses 224 annotated U2OS compounds and its own prespecified seed and candidate inventories.

The immutable release is available at https://doi.org/10.5281/zenodo.24008759. `scripts/fetch_data.py` retrieves the example and statistics inputs from that record; it does not download the full study collection. The historical full-study source and export records remain in the software ZIP.
