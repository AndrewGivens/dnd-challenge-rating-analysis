# D&D Challenge Rating Analysis

An undergraduate data acquisition and exploratory analysis project asking:

> Among monsters in selected third-party 5e source documents, how do listed challenge ratings compare with ratings predicted from hit points, armor class, and ability scores in the official 2014 SRD?

The analysis is exploratory. A statistical difference from the 2014 SRD baseline does not establish that a monster's rating is incorrect or unfair.

## Project status

This repository currently contains the acquisition workflow. The source coverage, usable sample size, missingness, and source-specific reuse terms still need to be reviewed before fixing the final scope or publishing data.

## Data source and method

The acquisition script uses Open5e's current v2 `/creatures/` endpoint. It follows the API's `next` pagination links, saves each raw JSON page locally, and writes a flattened CSV for inspection. The CSV includes source document and publisher metadata where provided, along with CR, HP, AC, and six ability scores.

Open5e aggregates material with different licenses. API availability does not by itself mean that all content can be republished. Review the license and attribution information for each source document before committing any downloaded data. The downloaded files are ignored by Git by default.

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
python scripts/acquire_open5e.py
```

To acquire one document only (for example, the 2024 SRD):

```bash
python scripts/acquire_open5e.py --document-key srd-2024
```

The script saves raw pages to `data/raw/open5e/` and the flattened table to `data/processed/creatures.csv`. These paths are git-ignored. Use `--help` to see options for a smaller feasibility pull, output directory, and request delay.

The script's output filename is `creatures.csv`; if you run it with a document filter, consider renaming or moving the output before acquiring another source so the first CSV is preserved. Review the resulting table for:

- counts by source document and publisher;
- missing challenge rating, hit points, and armor class;
- duplicates and variants;
- whether selected official and third-party sources have enough comparable records.

## Planned analysis

1. Define a comparison set with an explicit official baseline edition and selected third-party documents.
2. Describe coverage, missingness, CR distributions, and HP/AC relationships.
3. Fit a baseline model using official monsters only, then compare third-party listed CR with model predictions. Keep training and evaluation choices explicit, and interpret residuals as deviations from this baseline rather than judgments of game balance.
4. Investigate a small number of high-residual records and report data limitations.

Damage-per-round parsing is intentionally not part of the first acquisition pass. Attack descriptions vary in structure, so add a damage measure only after documenting a reliable parsing and validation approach.

## Repository layout

```text
scripts/acquire_open5e.py   API acquisition and flattening
data/raw/open5e/            locally cached API pages (ignored by Git)
data/processed/             locally generated CSV (ignored by Git)
```

## References

- [Open5e API documentation](https://open5e.com/api-docs)
- [Open5e source documents](https://open5e.com/sources)
- [Open5e API repository license notes](https://github.com/open5e/open5e-api/blob/staging/LICENSE.md)

## Reproducibility notes

The acquisition script records a retrieval timestamp and the API URL in each raw page. The API is maintained over time, so rerunning it can produce different records. Record the retrieval date and preserve the raw pages for the version used in the analysis. Do not commit API keys or downloaded content unless the applicable source terms permit redistribution.
