"""Acquire Open5e v2 creature data and save source-specific CSVs and raw pages."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


API_URL = "https://api.open5e.com/v2/creatures/"
USER_AGENT = "dnd-challenge-rating-analysis/0.2 (student data curation project)"
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
    "publisher_key",
    "publisher_name",
    "retrieved_at_utc",
]


def safe_name(value: str) -> str:
    """Make a document key safe for use in a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "unknown"


def display_name(value: Any) -> Any:
    """Flatten API objects such as {name, key} to their display name."""
    return value.get("name") if isinstance(value, dict) else value


def flatten_creature(creature: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    """Select analysis fields while retaining source and publisher provenance."""
    document = creature.get("document") or {}
    publisher = document.get("publisher") or {}
    abilities = creature.get("ability_scores") or {}

    return {
        "key": creature.get("key"),
        "name": creature.get("name"),
        "challenge_rating": creature.get("challenge_rating_decimal")
        if creature.get("challenge_rating_decimal") is not None
        else creature.get("challenge_rating"),
        "hit_points": creature.get("hit_points"),
        "armor_class": creature.get("armor_class"),
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
        "publisher_key": publisher.get("key"),
        "publisher_name": publisher.get("name"),
        "retrieved_at_utc": retrieved_at,
    }


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

    next_url: str | None = API_URL
    page_number = 1
    pages_fetched = 0
    creatures: list[dict[str, Any]] = []
    api_total: int | None = None

    while next_url and (max_pages is None or page_number <= max_pages):
        params = {"limit": limit}
        if document_key:
            params["document__key"] = document_key
        response = session.get(next_url, params=params if page_number == 1 else None, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError(f"Unexpected API response on page {page_number}: expected a results list")

        api_total = payload.get("count", api_total)
        pages_fetched += 1
        page_path = raw_dir / f"page_{page_number:04d}.json"
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
            if urlparse(next_url).hostname != "api.open5e.com":
                raise ValueError(f"Unexpected pagination host: {urlparse(next_url).hostname}")
            page_number += 1
            if delay:
                time.sleep(delay)

    rows = [flatten_creature(creature, retrieved_at) for creature in creatures]
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    complete = next_url is None
    suffix = "_partial" if not complete else ""
    csv_path = processed_dir / f"creatures_{label}{suffix}.csv"
    manifest_path = processed_dir / f"creatures_{label}{suffix}.manifest.json"
    frame.to_csv(csv_path, index=False)

    manifest = {
        "api_endpoint": API_URL,
        "document_key_filter": document_key,
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
    parser.add_argument("--delay", type=float, default=0.2, help="pause between page requests in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="data directory (default: data)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    document_keys = list(dict.fromkeys(args.document_key or [None]))
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
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
        print(f"Saved {manifest['csv_path']}")


if __name__ == "__main__":
    main()
