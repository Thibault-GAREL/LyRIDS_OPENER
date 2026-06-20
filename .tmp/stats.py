import json, collections
sets = ["AI","Lit","Mus","Pol","Sci","WNUT","Rest","Movie","Fab","Bio","CoNLL","GUM","GEN"]
# OPENER-Sup gold (tab:et-gold) and e2e (tab:main)
gold = [57.6,67.0,71.8,74.8,68.9,38.6,63.0,66.9,59.2,71.2,71.5,50.3,51.5]
e2e  = [40.2,49.6,55.3,50.3,51.7,24.9,33.6,37.8,30.2,36.5,44.8,33.7,34.2]
gap = [(s, round(g-e,1)) for s,g,e in zip(sets,gold,e2e)]
print("=== gold - e2e gap (OPENER-Sup), trie desc (detection penalty) ===")
for s,d in sorted(gap, key=lambda x:-x[1]): print(f"  {s:6} {d}")
print(f"  mean gap = {round(sum(g-e for g,e in zip(gold,e2e))/13,1)}")
# correlation gap vs whether specialised (MIT/Fab/Bio)
spec = {"Rest","Movie","Fab","Bio"}
print(f"  mean gap specialised(Rest/Movie/Fab/Bio) = {round(sum(d for s,d in gap if s in spec)/4,1)}")
print(f"  mean gap encyclopedic(AI/Lit/Mus/Pol/Sci) = {round(sum(d for s,d in gap if s in {'AI','Lit','Mus','Pol','Sci'})/5,1)}")

# confused type pairs from hard-mining cache
import os
for f in ["outputs/cache/hard_triplets_big.json","outputs/cache/hard_triplets.json"]:
    if os.path.exists(f):
        print(f"\n=== confused pairs from {f} ===")
        data = json.load(open(f, encoding="utf-8"))
        print("  type:", type(data).__name__, "| len:", len(data) if hasattr(data,'__len__') else '?')
        # peek structure
        sample = data[0] if isinstance(data, list) else data
        print("  sample keys/type:", list(sample.keys()) if isinstance(sample, dict) else str(sample)[:200])
        break
