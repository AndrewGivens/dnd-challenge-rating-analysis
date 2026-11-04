# D&D Challenge Rating Analysis

An undergraduate data acquisition and exploratory analysis project asking:

> Among monsters in selected third-party 5e source documents, how do listed challenge ratings compare with ratings predicted from hit points, armor class, and ability scores in the official 2014 SRD?

The analysis is exploratory. A statistical difference from the 2014 SRD baseline does not establish that a monster's rating is incorrect or unfair.

## Project status

This repository contains a reproducible acquisition workflow and data-quality diagnostics. The final source set and source-specific reuse terms still need review before publishing a dataset or fixing the analysis scope.

## Data source and method

The acquisition script uses Open5e's current v2 `/creatures/` endpoint. It follows the API's `next` pagination links, saves each raw JSON page locally, and writes a flattened CSV for inspection. The CSV includes source document and publisher metadata where provided, along with CR, HP, AC, and six ability scores.

Open5e aggregates material with different licenses. The API's license metadata is a starting point for review, not a blanket redistribution grant. Check the source's license and attribution requirements before committing downloaded content. The downloaded files are ignored by Git by default.

## Set up

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## Acquire data

```bash
python scripts/acquire_open5e.py --document-key srd-2014 --document-key ccdx
```

Each `--document-key` is acquired separately. The script follows pagination with a one-second pause between page requests, saves raw response pages in a timestamped folder, and writes a flattened CSV plus a manifest for each document. It also retrieves that document's Open5e license metadata and records license keys/names in the manifest and CSV. Unfiltered acquisition (the default) does not perform per-document license lookups; use selected document keys for a final, reviewable dataset. For a small feasibility pull, cap pages:

```bash
python scripts/acquire_open5e.py --document-key srd-2024 --max-pages 1
```

Complete pulls are named `data/processed/creatures_<document-key>.csv` with a matching `.manifest.json`. Incomplete pulls made with `--max-pages` get a `_partial` suffix so they do not overwrite complete data. Raw pages are saved beneath `data/raw/open5e/<document-key>/<run-timestamp>/`. These paths are git-ignored.

Review source coverage, missing values, duplicates, and numeric summaries across chosen sources:

```bash
python scripts/validate_data.py data/processed/creatures_srd-2014.csv data/processed/creatures_ccdx.csv --report data/processed/validation.json
```

The validator checks the 200 observation target, key uniqueness, repeated names within a source, numeric fields, source, publisher, and license counts, missingness, and whether each acquisition manifest indicates a complete pull. It reports warnings without deleting or deduplicating records; review flagged cases and document cleaning decisions.

Before adding a damage feature, inspect how much of the source has structured attack information:

```bash
python scripts/inspect_attack_coverage.py --raw-dir data/raw/open5e --report data/processed/attack_coverage.json
```

This diagnostic screens action descriptions, counts structured attack and damage fields and save-based damage text, and shows examples where attack-like language lacks structured fields. Text matching is only a screening cue; review examples manually. It does not calculate damage per round.

## Planned analysis

1. Define a comparison set with an explicit official baseline edition and selected third-party documents.
2. Describe coverage, missingness, CR distributions, and HP/AC relationships.
3. Fit a baseline model using official monsters only, then compare third-party listed CR with model predictions. Keep training and evaluation choices explicit, and interpret residuals as deviations from this baseline rather than judgments of game balance.
4. Investigate a small number of high-residual records and report data limitations.

Damage-per-round parsing is intentionally not part of the first analysis pass. The earlier exploration found missing or inconsistent structured attack fields and save-based damage embedded in prose. Add a damage measure only after defining “per round,” validating the parser against reviewed examples, and documenting cases it cannot interpret.

## Repository layout

```text
scripts/acquire_open5e.py          paginated API acquisition and flattening
scripts/validate_data.py           coverage and data-quality report
scripts/inspect_attack_coverage.py attack-field diagnostics from cached raw pages
data/raw/open5e/                   locally cached API pages (ignored by Git)
data/processed/                    CSVs, manifests, and reports (ignored by Git)
```

## References

- [Open5e API documentation](https://open5e.com/api-docs)
- [Open5e source documents](https://open5e.com/sources)
- [Open5e API repository license notes](https://github.com/open5e/open5e-api/blob/staging/LICENSE.md)

## Reproducibility notes

The acquisition script records a retrieval timestamp and the API URL in each raw page. The API is maintained over time, so rerunning it can produce different records. Record the retrieval date and preserve the raw pages for the version used in the analysis. Do not commit API keys or downloaded content unless the applicable source terms permit redistribution.
