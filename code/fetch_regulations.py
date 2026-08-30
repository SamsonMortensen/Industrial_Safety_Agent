"""Download the federal safety regulations governing rail-served intermodal yard operations.

The compliance auditor operates under two primary federal safety titles:
  - OSHA General Industry: Title 29 CFR Part 1910
  - FRA Railroad Safety: Title 49 CFR Part 228

Only the specific subparts an intermodal yard operates under are downloaded,
which keeps the corpus focused at roughly 750 KB instead of dozens of megabytes
of irrelevant law:

  29 CFR 1910 Subpart D  Walking-Working Surfaces  (housekeeping, spills, slip hazards)
  29 CFR 1910 Subpart N  Materials Handling and Storage  (powered industrial trucks, cranes)
  29 CFR 1910 Subpart S  Electrical  (clearances from energized parts)
  49 CFR 228             Hours of Service  (operator fatigue, duty limits, recordkeeping)

Each chunk carries the citation it came from, which is what lets the audit step
verify that a model-supplied section number is real instead of taking it on
faith.

    python code/fetch_regulations.py

Source: eCFR versioner API, https://www.ecfr.gov/developers/documentation/api/v1
"""

import json
import re
import sys
import time
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

API = "https://www.ecfr.gov/api/versioner/v1/full/{edition}/title-{title}.xml"
EDITION = "2025-01-01"
OUT = Path(__file__).resolve().parent.parent / "json" / "regulations.json"

SOURCES = [
    {
        "title": "29",
        "part": "1910",
        "subpart": "D",
        "subtitle": "B",
        "chapter": "XVII",
        "topic": "Walking-Working Surfaces",
    },
    {
        "title": "29",
        "part": "1910",
        "subpart": "N",
        "subtitle": "B",
        "chapter": "XVII",
        "topic": "Materials Handling and Storage",
    },
    {
        "title": "29",
        "part": "1910",
        "subpart": "S",
        "subtitle": "B",
        "chapter": "XVII",
        "topic": "Electrical",
    },
    {
        "title": "49",
        "part": "228",
        "subtitle": "B",
        "chapter": "II",
        "topic": "Hours of Service",
    },
]

MAX_CHARS = 1800


def fetch(src):
    params = {
        k: v for k, v in src.items() if k in ("subtitle", "chapter", "part", "subpart")
    }
    url = API.format(edition=EDITION, title=src["title"])
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    return r.content


def clean(text):
    text = text.replace("§", "Sec.")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split(text, limit=MAX_CHARS):
    """Break on paragraph markers -- (a), (b), (1) -- before falling back to length."""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for piece in re.split(r"(?=\((?:[a-z]|\d{1,2})\)\s)", text):
        if len(buf) + len(piece) > limit and buf:
            parts.append(buf.strip())
            buf = piece
        else:
            buf += piece
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if p]


def parse(xml, src):
    root = ET.fromstring(xml)
    chunks = []
    for div in root.iter():
        if not div.tag.startswith("DIV") or div.get("TYPE") != "SECTION":
            continue
        section = (div.get("N") or "").strip()
        if not section:
            continue
        head_el = div.find("HEAD")
        heading = clean("".join(head_el.itertext())) if head_el is not None else ""
        heading = re.sub(r"^Sec\.\s*[\d.]+\s*", "", heading).strip()

        body = clean(" ".join("".join(p.itertext()) for p in div.iter("P")))
        if len(body) < 120:
            continue

        citation = f"{src['title']} CFR {section}"
        for part_text in split(body):
            chunks.append(
                {
                    "citation": citation,
                    "section": section,
                    "heading": heading,
                    "title": src["title"],
                    "part": src["part"],
                    "subpart": src.get("subpart"),
                    "topic": src["topic"],
                    "text": part_text,
                    "char_len": len(part_text),
                }
            )
    return chunks


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for src in SOURCES:
        label = f"{src['title']} CFR {src['part']}" + (
            f" Subpart {src['subpart']}" if src.get("subpart") else ""
        )
        print(f"Fetching {label} ({src['topic']})...", flush=True)
        chunks = parse(fetch(src), src)
        sections = sorted({c["section"] for c in chunks})
        print(f"  {len(chunks)} chunks across {len(sections)} sections")
        all_chunks.extend(chunks)
        time.sleep(0.3)

    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i

    doc = {
        "generated": date.today().isoformat(),
        "edition": EDITION,
        "source": "eCFR versioner API",
        "sources": [{k: v for k, v in s.items() if k != "subtitle"} for s in SOURCES],
        "total_chunks": len(all_chunks),
        "sections": sorted({c["section"] for c in all_chunks}),
        "chunks": all_chunks,
    }
    OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")

    size = OUT.stat().st_size / 1000
    print(
        f"\nWrote {OUT.name}: {len(all_chunks)} chunks, "
        f"{len(doc['sections'])} sections, {size:.0f} KB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
