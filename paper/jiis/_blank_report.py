"""Blank-space report for main.pdf: lists the pages where a column ends well above the
bottom of the text area (large white gap). Run after _compile.ps1: python _blank_report.py
"""
import fitz

THRESHOLD = 0.85  # a column filled below 85% of the text height is reported

doc = fitz.open("main.pdf")
# text area of sn-jnl iicol: taken from the body pages themselves (min top / max bottom of blocks)
tops, bottoms = [], []
for page in doc:
    blocks = [b for b in page.get_text("blocks") if b[4].strip() and not b[4].strip().isdigit()]
    if blocks:
        tops.append(min(b[1] for b in blocks))
        bottoms.append(max(b[3] for b in blocks))
top, bottom = sorted(tops)[len(tops) // 2], max(bottoms)
height = bottom - top

last = len(doc)
print(f"{'page':>4s}  left col  right col")
for i, page in enumerate(doc, start=1):
    if i == last:
        continue  # the last page is naturally partial
    mid = page.rect.width / 2
    items = [b for b in page.get_text("blocks") if b[4].strip() and not b[4].strip().isdigit()]
    items += [(r.x0, r.y0, r.x1, r.y1, "img") for r in (page.get_image_rects(x[0])[0] for x in page.get_images()
                                                          if page.get_image_rects(x[0]))]
    for d in page.get_drawings():
        r = d["rect"]
        if r.height < 400:
            items.append((r.x0, r.y0, r.x1, r.y1, "draw"))
    fill = []
    for side in ("left", "right"):
        col = [b for b in items if (b[0] < mid - 5 if side == "left" else b[2] > mid + 5)]
        low = max((b[3] for b in col), default=top)
        fill.append((low - top) / height)
    if min(fill) < THRESHOLD:
        print(f"{i:>4d}  {fill[0]:7.0%}  {fill[1]:8.0%}")
