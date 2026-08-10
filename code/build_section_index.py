"""Build the list of section numbers that actually exist in 29 CFR and 49 CFR.

The audit checks whether a citation the model produced is real. Checking it
against the retrieved corpus alone would be unfair to the ungrounded baseline: a
model citing 29 CFR 1910.147 (lockout/tagout) is citing a real rule, just one
outside the four subparts this project retrieves over. Scoring that as
fabricated would overstate the benefit of retrieval.

So existence is checked against every section in Titles 29 and 49, taken from
the eCFR structure endpoint, and groundedness is tracked as a separate question.

    python code/build_section_index.py
"""
import json
import sys

from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent.parent / "json" / "valid_sections.json"
EDITION = "2025-01-01"
TITLES = ["29", "49"]


def sections_in(title):
    r = requests.get(
        f"https://www.ecfr.gov/api/versioner/v1/structure/{EDITION}/title-{title}.json",
        timeout=300)
    r.raise_for_status()

    found = set()

    def walk(node):
        if node.get("type") == "section":
            ident = (node.get("identifier") or "").strip()
            if ident:
                found.add(ident)
        for child in node.get("children") or []:
            walk(child)

    walk(r.json())
    return found


def main():
    index = {}
    everything = set()
    for title in TITLES:
        print(f"Reading structure of {title} CFR...", flush=True)
        found = sections_in(title)
        index[title] = sorted(found)
        everything |= found
        print(f"  {len(found):,} sections")

    OUT.write_text(json.dumps({
        "edition": EDITION,
        "source": "eCFR structure API",
        "titles": TITLES,
        "total_sections": len(everything),
        "by_title": index,
    }, indent=1), encoding="utf-8")
    print(f"\nWrote {OUT.name}: {len(everything):,} real section numbers "
          f"({OUT.stat().st_size/1000:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
