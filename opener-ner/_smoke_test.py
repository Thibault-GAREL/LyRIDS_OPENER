"""Smoke test end-to-end du package opener-ner (ZS + Sup) sur l'embedder local."""
import time
from opener import OpenerZS, OpenerSup

EMB = "outputs/models/embedder_contrastive_hard_big"   # chemin local (relatif a la racine)


def show(title, ents):
    print(f">>> {title}")
    if not ents:
        print("   (aucune entite)")
    for e in ents:
        print(f"   [{e['label']:14}] {e['text']!r}  (score {e['score']:.2f})")


print(">>> Chargement OPENER-ZS (embedder + GLiNER-L)...", flush=True)
t = time.time()
zs = OpenerZS.from_pretrained(EMB)
print(f"   charge en {time.time()-t:.1f}s", flush=True)

text = "Marie Curie discovered radium at the University of Paris."
labels = ["person", "discovery", "organization", "location"]
show(f"ZS predict | labels={labels}\n    texte: {text}", zs.predict(text, labels))

print("\n>>> Chargement OPENER-Sup + fit sur 3 exemples...", flush=True)
sup = OpenerSup.from_pretrained(EMB)
train_texts = [
    "Marie Curie discovered radium.",
    "Albert Einstein formulated relativity in Bern.",
    "Tokyo hosted the Olympic Games.",
]
train_annot = [
    [(0, 11, "person"), (23, 29, "substance")],
    [(0, 15, "person"), (41, 45, "location")],
    [(0, 5, "location")],
]
sup.fit(train_texts, train_annot)
print(f"   tete entrainee | classes = {sup.labels_}", flush=True)
show("Sup predict | texte: Niels Bohr worked in Copenhagen.",
     sup.predict("Niels Bohr worked in Copenhagen."))

print("\nOK - smoke test termine.")
