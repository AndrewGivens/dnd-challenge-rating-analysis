"""Report coverage, duplication, completeness, and numeric issues in acquired CSVs."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "key",
    "name",
    "challenge_rating",
    "hit_points",
    "armor_class",
    "strength",
    "dexterity",
    "constitution",
    "intelligence",
    "wisdom",
    "charisma",
    "document_key",
]
NUMERIC_COLUMNS = [
    "challenge_rating",
    "hit_points",
    "armor_class",
    "strength",
    "dexterity",
    "constitution",
    "intelligence",
    "wisdom",
    "charisma",
]


def parse_number(value: Any, *, allow_fraction: bool = False) -> float:
    """Parse a numeric API value, optionally accepting CR strings like '1/2'."""
    if pd.isna(value):
        return float("nan")
    try:
        # Some CRs are written as fractions such as "1/2"; other fields are
        # expected to be ordinary numeric values.
        if allow_fraction and isinstance(value, str) and "/" in value:
            return float(Fraction(value.strip()))
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def load_inputs(paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read CSV inputs and check adjacent acquisition manifests when available."""
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["_input_file"] = str(path)
        frames.append(frame)

        # Acquisition writes a manifest beside each complete CSV. Reading it
        # here lets validation compare the saved rows with the API's reported count.
        manifest_path = path.with_name(f"{path.stem}.manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["manifest_path"] = str(manifest_path)
            manifest["csv_rows"] = len(frame)
            manifests.append(manifest)
    if not frames:
        raise ValueError("Provide at least one input CSV")
    return pd.concat(frames, ignore_index=True, sort=False), manifests


def summarize(frame: pd.DataFrame, manifests: list[dict[str, Any]], minimum_rows: int) -> dict[str, Any]:
    # This report flags issues for review; it deliberately does not drop or
    # rewrite records, because duplicates and unusual values need human review.
    errors: list[str] = []
    warnings: list[str] = []
    # Keep only expected fields that are absent from the combined table.
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"Required columns missing: {', '.join(missing_columns)}")

    row_count = len(frame)
    if row_count == 0:
        errors.append("Dataset has no rows")
    elif row_count < minimum_rows:
        warnings.append(f"Only {row_count} observations; assignment target is at least {minimum_rows}")

    # isna() marks missing cells; sum counts them and mean gives the fraction.
    missingness = {
        column: {
            "missing_count": int(frame[column].isna().sum()),
            "missing_percent": round(float(frame[column].isna().mean() * 100), 2),
        }
        for column in REQUIRED_COLUMNS
        if column in frame.columns
    }
    identity_fields = [column for column in ("key", "name", "document_key") if column in frame.columns]
    missing_identity = {
        column: int(frame[column].isna().sum()) for column in identity_fields if frame[column].isna().any()
    }
    if missing_identity:
        warnings.append(f"Missing values in identity/provenance fields: {missing_identity}")

    numeric_summary: dict[str, Any] = {}
    invalid_numeric: dict[str, int] = {}
    numeric_frame = pd.DataFrame(index=frame.index)
    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        allow_fraction = column == "challenge_rating"
        # Parse into a separate numeric view so the source CSV remains untouched.
        # Series.map applies the parser to every value; the lambda passes the
        # column-specific rule for fractional challenge ratings.
        converted = frame[column].map(lambda value: parse_number(value, allow_fraction=allow_fraction))
        invalid_count = int((frame[column].notna() & converted.isna()).sum())
        if invalid_count:
            invalid_numeric[column] = invalid_count
            warnings.append(f"{invalid_count} nonnumeric values in {column}")
        numeric_frame[column] = converted
        values = converted.dropna()
        numeric_summary[column] = {
            "n": int(values.size),
            "min": float(values.min()) if not values.empty else None,
            "median": float(values.median()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        }

    range_issues: dict[str, int] = {}
    for column in NUMERIC_COLUMNS:
        if column not in numeric_frame:
            continue
        values = numeric_frame[column]
        # Negative CR, HP, or ability scores are implausible. AC must be positive.
        if column in ("challenge_rating", "hit_points", "strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"):
            bad_count = int((values < 0).sum())
        elif column == "armor_class":
            bad_count = int((values <= 0).sum())
        else:
            bad_count = 0
        if bad_count:
            range_issues[column] = bad_count
            warnings.append(f"{bad_count} implausible or out-of-range values in {column}")

    duplicate_key_rows = 0
    if "key" in frame.columns:
        # duplicated() returns True/False per row; keep=False marks every row
        # participating in a repeat, not only the extra copies. sum() counts them.
        duplicate_key_rows = int(frame["key"].duplicated(keep=False).sum())
        if duplicate_key_rows:
            warnings.append(f"{duplicate_key_rows} rows have a duplicated API key; inspect before analysis")

    duplicate_name_source_rows = 0
    if {"name", "document_key"}.issubset(frame.columns):
        # A name can legitimately recur in different books, so only flag a name
        # repeated within the same source document.
        duplicate_name_source_rows = int(
            frame.duplicated(subset=["name", "document_key"], keep=False).sum()
        )
        if duplicate_name_source_rows:
            warnings.append(
                f"{duplicate_name_source_rows} rows repeat a name within a source document; "
                "review variants rather than dropping them automatically"
            )

    source_counts: list[dict[str, Any]] = []
    if "document_key" in frame.columns:
        # value_counts groups identical source names; this comprehension turns
        # that result into a list of dictionaries for the JSON report.
        source_counts = [
            {"document_key": str(source), "count": int(count)}
            for source, count in frame["document_key"].fillna("(missing)").value_counts().items()
        ]
        if len(source_counts) < 2:
            warnings.append("Fewer than two source documents are represented")

    publisher_counts: list[dict[str, Any]] = []
    if "publisher_name" in frame.columns:
        publisher_counts = [
            {"publisher_name": str(source), "count": int(count)}
            for source, count in frame["publisher_name"].fillna("(missing)").value_counts().items()
        ]

    license_counts: dict[str, int] = {}
    missing_license_rows = 0
    if "document_license_keys" in frame.columns:
        missing_license_rows = int(frame["document_license_keys"].fillna("").str.strip().eq("").sum())
        for value in frame["document_license_keys"].dropna():
            for license_key in str(value).split(";"):
                if license_key:
                    license_counts[license_key] = license_counts.get(license_key, 0) + 1
        if missing_license_rows:
            warnings.append(
                f"{missing_license_rows} rows lack API license metadata; check the source documents manually"
            )

    # A manifest can reveal a truncated pull even when the resulting CSV looks
    # internally consistent, so check completion and row totals as well.
    for manifest in manifests:
        expected = manifest.get("api_total_reported")
        acquired = manifest.get("records_acquired")
        if not manifest.get("complete", False):
            warnings.append(f"Acquisition was partial: {manifest.get('manifest_path')}")
        if manifest.get("document_key_filter") and not manifest.get("licenses"):
            warnings.append(f"No license metadata in acquisition manifest: {manifest.get('manifest_path')}")
        if expected is not None and acquired != expected:
            warnings.append(
                f"Acquisition count {acquired} differs from API total {expected}: "
                f"{manifest.get('manifest_path')}"
            )

    return {
        "status": "ERROR" if errors else "WARN" if warnings else "PASS",
        "rows": row_count,
        "minimum_rows_target": minimum_rows,
        "input_files": sorted(frame["_input_file"].dropna().unique().tolist()) if "_input_file" in frame else [],
        "source_document_count": len(source_counts),
        "source_counts": source_counts,
        "publisher_counts": publisher_counts,
        "license_key_counts": license_counts,
        "missing_license_rows": missing_license_rows,
        "missingness": missingness,
        "missing_identity_values": missing_identity,
        "numeric_summary": numeric_summary,
        "invalid_numeric_values": invalid_numeric,
        "range_issues": range_issues,
        "duplicate_key_rows": duplicate_key_rows,
        "duplicate_name_source_rows": duplicate_name_source_rows,
        "manifests": manifests,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="one or more acquired CSVs to inspect")
    parser.add_argument(
        "--minimum-rows",
        type=int,
        default=200,
        help="minimum observation count to report (assignment default: 200)",
    )
    parser.add_argument("--report", type=Path, help="optional path for a JSON report")
    args = parser.parse_args()
    if args.minimum_rows < 1:
        parser.error("--minimum-rows must be at least 1")

    try:
        frame, manifests = load_inputs(args.csv)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    report = summarize(frame, manifests, args.minimum_rows)
    print(json.dumps(report, indent=2, allow_nan=False))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved report: {args.report}")

    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
