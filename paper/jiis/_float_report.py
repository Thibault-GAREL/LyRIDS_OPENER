"""Float placement report: for each figure and table of main.pdf, the page of its first
mention in the text versus the page where the float itself is printed.
Run after _compile.ps1: python _float_report.py
"""
import re

import fitz

doc = fitz.open("main.pdf")
pages = [p.get_text() for p in doc]

floats = []
for kind, cap_re, ref_word in [("Table", r"(?m)^Table (\d+) [A-Z]", "Table"), ("Fig.", r"(?m)^Fig\. (\d+) [A-Z]", "Figure")]:
    seen = set()
    for i, txt in enumerate(pages):
        for m in re.finditer(cap_re, txt):
            n = int(m.group(1))
            if n in seen:
                continue
            seen.add(n)
            # a mention is "Table N" / "Figure N" that is not the caption label itself
            mention = rf"(?<!^)(?<!\n){ref_word}\s+{n}\b" if kind == "Table" else rf"{ref_word}\s+{n}\b"
            first = next((j for j, t in enumerate(pages) if re.search(mention, t)), None)
            floats.append((kind, n, i + 1, first + 1 if first is not None else None))

print(f"{'float':10s} {'cited p.':>8s} {'printed p.':>10s} {'gap':>4s}")
for kind, n, printed, cited in sorted(floats, key=lambda x: (x[3] or 0, x[0])):
    gap = printed - cited if cited else None
    print(f"{kind + ' ' + str(n):10s} {cited!s:>8s} {printed:>10d} {gap!s:>4s}")
print(f"pages: {len(doc)}")
