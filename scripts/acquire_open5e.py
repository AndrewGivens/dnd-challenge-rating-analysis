"""Download Open5e v2 creature records and create a flattened CSV.

Downloaded game content is written locally and ignored by Git. Check the
license and attribution terms for each source document before redistributing it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


API_URL = "https://api.open5e.com/v2/creatures/"
USER_AGENT = "dnd-challenge-rating-analysis/0.1 (student data curation project)"


def flatten_creature(creature: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    """Keep analysis fields and source provenance from one API creature."""
    document = creature.get("document") or {}
    publisher = document.get("publisher") or {}
    ability_scores = creature.get("ability_scores") or {}
    cr = creature.get("challenge_rating_decimal")
    if cr is None:
        cr = creature.get("challenge_rating")

    def display_name(value: Any) -> Any:
        return value.get("name") if isinstance(value, dict) else value

    return {
        "key": creature.get("key"),
        "name": creature.get("name"),
        "challenge_rating": cr,
        "hit_points": creature.get("hit_points"),
        "armor_class": creature.get("armor_class"),
        "strength": ability_scores.get("strength", creature.get("strength")),
        "dexterity": ability_scores.get("dexterity", creature.get("dexterity")),
        "constitution": ability_scores.get("constitution", creature.get("constitution")),
        "intelligence": ability_scores.get("intelligence", creature.get("intelligence")),
        "wisdom": ability_scores.get("wisdom", creature.get("wisdom")),
        "charisma": ability_scores.get("charisma", creature.get("charisma")),
        "size": display_name(creature.get("size")),
        "type": display_name(creature.get("type")),
        "subtype": creature.get("subtype"),
        "alignment": creature.get("alignment"),
        "document_key": document.get("key"),
        "document_name": document.get("name"),
        "document_display_name": document.get("display_name"),
        "document_type": document.get("type"),
        "document_permalink": document.get("permalink"),
        "publisher_key": publisher.get("key"),
        "publisher_name": publisher.get("name"),
        "retrieved_at_utc": retrieved_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="records per API page (default: 100)")
    parser.add_argument("--max-pages", type=int, help="maximum pages to fetch; useful for a feasibility pull")
    parser.add_argument("--document-key", help="fetch only one source document, such as srd-2024")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds to wait between requests (default: 0.2)")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="root data directory (default: data)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    raw_dir = args.output_dir / "raw" / "open5e"
    processed_dir = args.output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    next_url: str | None = API_URL
    page_number = 1
    all_creatures: list[dict[str, Any]] = []
    expected_count: int | None = None

    while next_url and (args.max_pages is None or page_number <= args.max_pages):
        params = {"limit": args.limit} if page_number == 1 else None
        if page_number == 1 and args.document_key:
            params["document__key"] = args.document_key
        response = session.get(next_url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("Unexpected API response: expected an object with a results list")

        expected_count = payload.get("count", expected_count)
        page_path = raw_dir / f"page_{page_number:04d}.json"
        page_path.write_text(
            json.dumps(
                {
                    "retrieved_at_utc": retrieved_at,
                    "request_url": response.url,
                    "results": payload["results"],
                    "count": payload.get("count"),
                    "next": payload.get("next"),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        all_creatures.extend(payload["results"])
        print(f"Fetched page {page_number}: {len(payload['results'])} records")
        next_url = payload.get("next")
        if next_url:
            # Follow only Open5e links; fail safely if the API unexpectedly
            # returns a link to another host.
            if urlparse(next_url).hostname != "api.open5e.com":
                raise ValueError(f"Unexpected pagination host: {urlparse(next_url).hostname}")
            page_number += 1
            if args.delay:
                time.sleep(args.delay)

    flattened = pd.DataFrame(flatten_creature(item, retrieved_at) for item in all_creatures)
    csv_path = processed_dir / "creatures.csv"
    flattened.to_csv(csv_path, index=False)

    print(f"API total reported: {expected_count}")
    print(f"Records acquired: {len(flattened)}")
    print(f"Unique keys: {flattened['key'].nunique() if 'key' in flattened else 0}")
    print(f"Sources represented: {flattened['document_key'].nunique() if 'document_key' in flattened else 0}")
    for column in ("challenge_rating", "hit_points", "armor_class"):
        if column in flattened:
            missing_pct = flattened[column].isna().mean() * 100
            print(f"Missing {column}: {missing_pct:.1f}%")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved raw pages: {raw_dir}")
    if args.max_pages:
        print("Note: this was a partial pull because --max-pages was set.")


if __name__ == "__main__":
    main()
