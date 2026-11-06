"""Acquire Open5e v2 creature data and save source-specific CSVs and raw pages."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pandas as pd
import requests


API_URL = "https://api.open5e.com/v2/creatures/"
DOCUMENTS_URL = "https://api.open5e.com/v2/documents/"
# Identify this student project to the API operator; do not impersonate a browser.
USER_AGENT = "dnd-challenge-rating-analysis/0.2 (student data curation project)"
# Keep column order stable so CSVs are predictable across different source books.
OUTPUT_COLUMNS = [
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
    "size",
    "type",
    "subtype",
    "alignment",
    "document_key",
    "document_name",
    "document_display_name",
    "document_permalink",
    "document_license_keys",
    "document_license_names",
    "publisher_key",
    "publisher_name",
    "retrieved_at_utc",
]


def safe_name(value: str) -> str:
    """Replace filename-unfriendly characters in an API document key."""
    # The regex allows letters, numbers, dots, underscores, and hyphens; each
    # run of other characters becomes one underscore. Strip edge punctuation,
    # and use a fallback if the input has no usable characters.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def display_name(value: Any) -> Any:
    """Flatten API objects such as {name, key} to their display name."""
    # Conditional expression: nested API objects become their name; other
    # values (including missing values) pass through unchanged.
    return value.get("name") if isinstance(value, dict) else value


def flatten_creature(
    creature: dict[str, Any], retrieved_at: str, document_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Extract selected nested fields into one flat CSV row with provenance."""
    document = creature.get("document") or {}
    publisher = document.get("publisher") or {}
    abilities = creature.get("ability_scores") or {}
    # Prefer the dedicated document endpoint, then fall back to the creature record.
    licenses = (document_metadata or {}).get("licenses") or document.get("licenses") or []
    # Select only present license values, then join multiple licenses into one
    # CSV cell instead of expanding a creature into multiple rows.
    license_keys = [license.get("key") for license in licenses if license.get("key")]
    license_names = [license.get("name") for license in licenses if license.get("name")]

    return {
        "key": creature.get("key"),
        "name": creature.get("name"),
        # Decimal CR preserves fractional ratings; use the older field as fallback.
        "challenge_rating": creature.get("challenge_rating_decimal")
        if creature.get("challenge_rating_decimal") is not None
        else creature.get("challenge_rating"),
        "hit_points": creature.get("hit_points"),
        "armor_class": creature.get("armor_class"),
        # Scores are nested in v2; the second argument is a fallback for other shapes.
        "strength": abilities.get("strength", creature.get("strength")),
        "dexterity": abilities.get("dexterity", creature.get("dexterity")),
        "constitution": abilities.get("constitution", creature.get("constitution")),
        "intelligence": abilities.get("intelligence", creature.get("intelligence")),
        "wisdom": abilities.get("wisdom", creature.get("wisdom")),
        "charisma": abilities.get("charisma", creature.get("charisma")),
        "size": display_name(creature.get("size")),
        "type": display_name(creature.get("type")),
        "subtype": creature.get("subtype"),
        "alignment": creature.get("alignment"),
        "document_key": document.get("key"),
        "document_name": document.get("name"),
        "document_display_name": document.get("display_name"),
        "document_permalink": document.get("permalink"),
        "document_license_keys": ";".join(license_keys),
        "document_license_names": ";".join(license_names),
        "publisher_key": publisher.get("key"),
        "publisher_name": publisher.get("name"),
        "retrieved_at_utc": retrieved_at,
    }


def fetch_document_metadata(session: requests.Session, document_key: str) -> dict[str, Any]:
    """Get source metadata separately because license details live on documents."""
    url = f"{DOCUMENTS_URL}{quote(document_key, safe='')}/"
    response = session.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected document metadata response for {document_key}")
    return payload


def fetch_document(
    session: requests.Session,
    document_key: str | None,
    output_dir: Path,
    run_id: str,
    retrieved_at: str,
    limit: int,
    max_pages: int | None,
    delay: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch every page for one document and save a raw copy of each response."""
    label = safe_name(document_key or "all")
    raw_dir = output_dir / "raw" / "open5e" / label / run_id
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    # A separate timestamped raw folder keeps each acquisition reproducible and
    # prevents a later run from silently overwriting the original API responses.
    document_metadata = fetch_document_metadata(session, document_key) if document_key else {}

    next_url: str | None = API_URL
    page_number = 1
    pages_fetched = 0
    creatures: list[dict[str, Any]] = []
    api_total: int | None = None

    # The API supplies the next page URL. Follow it until it is null, unless the
    # caller requested a small feasibility pull with --max-pages.
    while next_url and (max_pages is None or page_number <= max_pages):
        # These become URL query parameters such as ?limit=100&document__key=tob.
        params = {"limit": limit}
        if document_key:
            params["document__key"] = document_key
        # Filters and page size are sent on page one; later URLs already include
        # the pagination state, so do not append the original query again.
        response = session.get(next_url, params=params if page_number == 1 else None, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"Unexpected API response on page {page_number}: expected a results list")

        api_total = payload.get("count", api_total)
        pages_fetched += 1
        page_path = raw_dir / f"page_{page_number:04d}.json"
        # Save the unflattened JSON before cleaning, so decisions can be audited
        # or the flattened CSV can be rebuilt without another API request.
        page_path.write_text(
            json.dumps(
                {
                    "retrieved_at_utc": retrieved_at,
                    "request_url": response.url,
                    "count": payload.get("count"),
                    "next": payload.get("next"),
                    "results": payload["results"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        creatures.extend(payload["results"])
        print(f"{label}: fetched page {page_number} ({len(payload['results'])} records)")

        next_url = payload.get("next")
        if next_url:
            # Only follow pagination links on the expected API host.
            if urlparse(next_url).hostname != "api.open5e.com":
                raise ValueError(f"Unexpected pagination host: {urlparse(next_url).hostname}")
            page_number += 1
            if delay:
                time.sleep(delay)

    # Turn each nested API dictionary into one flat row dictionary.
    rows = [flatten_creature(creature, retrieved_at, document_metadata) for creature in creatures]
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # A page-capped run is useful for checking feasibility, but must be labelled
    # partial so it cannot be mistaken for a complete dataset.
    complete = next_url is None
    suffix = "_partial" if not complete else ""
    csv_path = processed_dir / f"creatures_{label}{suffix}.csv"
    manifest_path = processed_dir / f"creatures_{label}{suffix}.manifest.json"
    frame.to_csv(csv_path, index=False)

    # The manifest records enough provenance to check counts and source terms
    # later without relying on memory or rerunning the acquisition.
    manifest = {
        "api_endpoint": API_URL,
        "document_key_filter": document_key,
        "document_name": document_metadata.get("name"),
        "document_display_name": document_metadata.get("display_name"),
        "publisher": document_metadata.get("publisher"),
        "licenses": document_metadata.get("licenses", []),
        "api_total_reported": api_total,
        "records_acquired": len(frame),
        "page_count": pages_fetched,
        "complete": complete,
        "retrieved_at_utc": retrieved_at,
        "csv_path": str(csv_path),
        "raw_pages_path": str(raw_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-key",
        action="append",
        help="document key to acquire; repeat for multiple separate source files",
    )
    parser.add_argument("--limit", type=int, default=100, help="records per page (default: 100)")
    parser.add_argument("--max-pages", type=int, help="cap pages per document for a partial feasibility pull")
    parser.add_argument("--delay", type=float, default=1.0, help="pause between page requests in seconds (default: 1)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="data directory (default: data)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    # Preserve command-line order while avoiding fetching the same source twice.
    # With no --document-key, None means request the API's unfiltered collection.
    document_keys = list(dict.fromkeys(args.document_key or [None]))
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # A Session reuses HTTP connections and applies the same headers to requests.
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    for document_key in document_keys:
        frame, manifest = fetch_document(
            session=session,
            document_key=document_key,
            output_dir=args.output_dir,
            run_id=run_id,
            retrieved_at=retrieved_at,
            limit=args.limit,
            max_pages=args.max_pages,
            delay=args.delay,
        )
        print(
            f"{document_key or 'all sources'}: {len(frame)} rows; "
            f"API reports {manifest['api_total_reported']}; complete={manifest['complete']}"
        )
        if manifest["licenses"]:
            print("Licenses: " + "; ".join(license["name"] for license in manifest["licenses"]))
        else:
            print("License metadata unavailable for this pull; review each document before redistribution.")
        print(f"Saved {manifest['csv_path']}")


if __name__ == "__main__":
    main()
