import re

# JIIS version: single main.tex (no \input), bib synced next to it by _compile.ps1.
cited = set()
txt = open("main.tex", encoding="utf-8").read()
txt = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("%"))
for m in re.finditer(r"\\cite\{([^}]*)\}", txt):
    for k in m.group(1).split(","):
        cited.add(k.strip())

bibkeys = [m.group(1).strip()
           for m in re.finditer(r"@\w+\{([^,]+),", open("../references.bib", encoding="utf-8").read())]
uncited = [k for k in bibkeys if k not in cited]
missing = [k for k in cited if k and k not in bibkeys]

print(f"Entrees bib : {len(bibkeys)} | citees : {len(bibkeys)-len(uncited)} | NON citees : {len(uncited)}")
print("\n=== NON citees (absentes des references) ===")
for k in uncited:
    print("  ", k)
print("\n=== \\cite vers une cle ABSENTE du bib (erreur) ===")
print("  ", missing or "(aucune - OK)")
