import json, collections, os
base = "external/OWNER/data/lyrids"
order = [("crossner_ai","AI"),("crossner_literature","Lit"),("crossner_music","Mus"),
         ("crossner_politics","Pol"),("crossner_science","Sci"),("wnut17","WNUT"),
         ("mit_restaurant","Rest"),("mit_movie","Movie"),("fabner","Fab"),
         ("bionlp2004","Bio"),("conll2003","CoNLL"),("gum","GUM"),("gentle","GEN")]
for folder, abbr in order:
    p = os.path.join(base, folder, "test.json")
    docs = json.load(open(p, encoding="utf-8"))["documents"]
    # cap at 1000 sentences (these are ~1 sentence/doc)
    cnt = collections.Counter()
    nsent = 0
    for d in docs:
        nsent += len(d["sentences"])
        for e in d["entities"]:
            cnt[e["type"]] += 1
        if nsent >= 1000:
            break
    tot = sum(cnt.values())
    items = cnt.most_common()
    pcts = [(t, 100*c/tot) for t,c in items]
    s = " ".join(f"{t}:{round(p)}" for t,p in pcts)
    print(f"{abbr}\t{len(items)}types\ttot={tot}\t{s}")
