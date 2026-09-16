"""Build jiis/references.bib = shared ../references.bib + verified DOIs of doi_overlay.csv.

JIIS asks for full DOI links in the reference list. The shared bib stays untouched
(it also feeds the IEEE and KBS versions). A DOI is only added to an entry that has
none, and every DOI in the overlay was verified (see its "source" column).
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
bib = (HERE.parent / "references.bib").read_text(encoding="utf-8")

with open(HERE / "doi_overlay.csv", encoding="utf-8") as f:
    overlay = {row["key"]: row["doi"] for row in csv.DictReader(f)}

added = 0
for key, doi in overlay.items():
    m = re.search(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,(.*?)\n\}", bib, re.S)
    if m is None:
        raise SystemExit(f"key not found in ../references.bib: {key}")
    if re.search(r"\bdoi\s*=", m.group(1)):
        continue
    body = m.group(1).rstrip()
    if not body.endswith(","):
        body += ","
    new_body = f"{body}\n  doi       = {{{doi}}}"
    bib = bib[:m.start(1)] + new_body + bib[m.end(1):]
    added += 1

(HERE / "references.bib").write_text(bib, encoding="utf-8")
total = len(re.findall(r"\bdoi\s*=", bib))
print(f"references.bib synced: {added} DOI added from doi_overlay.csv, {total} entries with a DOI")
