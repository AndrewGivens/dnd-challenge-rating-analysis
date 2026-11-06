"""Inspect raw action records before attempting to estimate monster damage."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# These phrases only identify descriptions worth screening; they do not prove
# that an action is a damaging attack or that its damage is fully structured.
# `|` means “or”; `\b` requires word boundaries so “to hit” is matched as a
# phrase. re.IGNORECASE also matches capitalization variants in descriptions.
ATTACK_WORDING = re.compile(r"attack roll|\bto hit\b|weapon attack|spell attack", re.IGNORECASE)
DICE_WORDING = re.compile(r"\b\d+d\d+\b", re.IGNORECASE)


def has_structured_attack(action: dict[str, Any]) -> bool:
    """Recognize the nested v2 attack form and the legacy direct fields."""
    # Open5e v2 nests attack details; older records may put fields directly on
    # the action. Accept both shapes while checking how often data is absent.
    attacks = action.get("attacks")
    if isinstance(attacks, list) and attacks:
        return True
    return action.get("attack_bonus") is not None or bool(action.get("damage_dice"))


def has_structured_damage(action: dict[str, Any]) -> bool:
    """Check whether structured fields appear to describe damage, not just a hit."""
    # A hit can have attack fields but no damage. Check that at least one nested
    # attack has a positive dice count or nonzero damage bonus.
    def positive(value: Any) -> bool:
        try:
            return float(value or 0) > 0
        except (TypeError, ValueError):
            return bool(value)

    def nonzero(value: Any) -> bool:
        try:
            return float(value or 0) != 0
        except (TypeError, ValueError):
            return bool(value)

    attacks = action.get("attacks")
    if isinstance(attacks, list) and attacks:
        return any(
            isinstance(attack, dict)
            and (
                (attack.get("damage_die_type") and positive(attack.get("damage_die_count")))
                or nonzero(attack.get("damage_bonus"))
            )
            for attack in attacks
        )
    return bool(action.get("damage_dice")) or action.get("damage_bonus") is not None


def inspect(raw_dir: Path, max_examples: int, document_keys: set[str] | None = None) -> dict[str, Any]:
    # Read cached raw responses rather than requesting the API again.
    files = sorted(raw_dir.rglob("page_*.json"))
    if document_keys:
        # Ignore legacy one-off caches at the root of raw_dir; selected pulls
        # are stored under <document-key>/<timestamp>/page_*.json.
        # Keep only pages stored under a requested document-key folder.
        files = [path for path in files if path.parent.parent.name in document_keys]
    if not files:
        raise FileNotFoundError(f"No cached page_*.json files under {raw_dir}")

    totals: Counter[str] = Counter()
    # defaultdict creates a Counter automatically the first time a source is seen.
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, str]] = []
    creature_count = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for creature in payload.get("results", []):
            document = creature.get("document") or {}
            # Prefer the record's source key; parents[1] is a folder-name fallback
            # for cached API pages that do not include a document object.
            source = document.get("key") or path.parents[1].name
            if document_keys and source not in document_keys:
                continue
            creature_count += 1
            actions = creature.get("actions") or []
            # Count save-based dice language independently: it is a common
            # damage path that does not look like an attack roll.
            for action in actions:
                if not isinstance(action, dict):
                    continue
                description = str(action.get("desc") or "")
                if "saving throw" in description.lower() and DICE_WORDING.search(description):
                    totals["save_based_damage_actions"] += 1
                    per_source[source]["save_based_damage_actions"] += 1
                if not ATTACK_WORDING.search(description):
                    continue

                totals["attack_like_actions"] += 1
                per_source[source]["attack_like_actions"] += 1
                if has_structured_attack(action):
                    totals["with_structured_attack_data"] += 1
                    per_source[source]["with_structured_attack_data"] += 1
                    if not has_structured_damage(action):
                        totals["structured_without_damage_fields"] += 1
                        per_source[source]["structured_without_damage_fields"] += 1
                else:
                    totals["missing_structured_attack_data"] += 1
                    per_source[source]["missing_structured_attack_data"] += 1
                    # Keep a small review sample so the heuristic can be checked
                    # manually instead of treating text matching as ground truth.
                    if len(examples) < max_examples:
                        examples.append(
                            {
                                "source": source,
                                "creature": str(creature.get("name", "")),
                                "action": str(action.get("name", "")),
                                "description_excerpt": re.sub(r"\s+", " ", description)[:260],
                            }
                        )

    attack_like = totals["attack_like_actions"]
    missing = totals["missing_structured_attack_data"]
    # Convert Counter objects to regular dicts so json.dumps can serialize them.
    return {
        "raw_pages": len(files),
        "creatures_inspected": creature_count,
        "action_totals": dict(totals),
        "structured_attack_missing_percent": round(100 * missing / attack_like, 2) if attack_like else None,
        "by_source": {source: dict(counts) for source, counts in sorted(per_source.items())},
        "examples_missing_structured_attack_data": examples,
        "note": (
            "Text matches are screening cues, not confirmed attack labels. Review examples manually; "
            "this report does not estimate damage per round."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/open5e"))
    parser.add_argument(
        "--document-key",
        action="append",
        help="limit inspection to selected source documents; repeat as needed",
    )
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--report", type=Path, help="optional path for a JSON report")
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples cannot be negative")

    try:
        report = inspect(args.raw_dir, args.max_examples, set(args.document_key or []))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
        print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
