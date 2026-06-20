"""Generate the 3-panel UMAP figure showing contrastive learning 'in action'.

Embeds the gold mentions of a HELD-OUT domain (WNUT-17, not the CoNLL contrastive
source) with three embedders and projects each to 2D with UMAP, coloured by gold type:
    1. Frozen Nomic v1.5            (off-the-shelf, types intermixed)
    2. + contrastive fine-tuning    (Stage 1, CoNLL triplets)
    3. + hard-negative mining       (Stage 2, selected embedder)

Output: paper/assets/umap_contrastive.{pdf,png}

Run from repo root with the pytorch_cuda_env venv. Deterministic (seed 42).
"""
import json
import random
import collections

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DATA = "external/OWNER/data/lyrids/wnut17/test.json"      # held-out domain
TASK_PREFIX = "classification: "                            # Nomic v1.5 task prefix
ENT_PRE, ENT_SUF = "[ENT]", "[/ENT]"                       # span-in-context markers
MAX_MENTIONS = 1000
SEED = 42

MODELS = {
    "Frozen Nomic v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "+ contrastive": "outputs/models/embedder_contrastive",
    "+ hard-neg. mining": "outputs/models/embedder_contrastive_hard_big",
}


def build_inputs(path):
    """Reconstruct span-in-context strings + gold types from OWNER-format JSON."""
    docs = json.load(open(path, encoding="utf-8"))["documents"]
    inputs, types = [], []
    for d in docs:
        for e in d["entities"]:
            sent = d["sentences"][e["sentence_idx"]]
            full = " ".join(sent)
            s_tok, e_tok = e["start_word_idx"], e["end_word_idx"]
            start = len(" ".join(sent[:s_tok])) + (1 if s_tok > 0 else 0)
            ent_text = " ".join(sent[s_tok:e_tok])
            end = start + len(ent_text)
            payload = full[:start] + ENT_PRE + " " + ent_text + " " + ENT_SUF + full[end:]
            inputs.append(TASK_PREFIX + payload)
            types.append(e["type"])
    return inputs, types


def embed(model_id, inputs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(model_id, trust_remote_code=True, device=dev)
    emb = m.encode(inputs, convert_to_numpy=True, normalize_embeddings=True,
                   batch_size=32, show_progress_bar=True)
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return emb


def main():
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(SEED)
    np.random.seed(SEED)

    inputs, types = build_inputs(DATA)
    if len(inputs) > MAX_MENTIONS:
        idx = sorted(random.sample(range(len(inputs)), MAX_MENTIONS))
        inputs = [inputs[i] for i in idx]
        types = [types[i] for i in idx]
    type_order = [t for t, _ in collections.Counter(types).most_common()]
    print(f"[umap] {len(inputs)} mentions | types={type_order}", flush=True)

    embs2d = {}
    for name, mid in MODELS.items():
        print(f"[umap] embedding with {name} ({mid})", flush=True)
        E = embed(mid, inputs)
        print(f"[umap]   reducing {E.shape} -> 2D", flush=True)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED)
        embs2d[name] = reducer.fit_transform(E)

    cmap = plt.get_cmap("tab10")
    color = {t: cmap(i % 10) for i, t in enumerate(type_order)}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, (name, xy) in zip(axes, embs2d.items()):
        for t in type_order:
            mask = np.array([ty == t for ty in types])
            ax.scatter(xy[mask, 0], xy[mask, 1], s=7, color=color[t], label=t,
                       alpha=0.75, linewidths=0)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(markerscale=2, fontsize=8, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig("paper/assets/umap_contrastive.pdf", bbox_inches="tight")
    fig.savefig("paper/assets/umap_contrastive.png", dpi=150, bbox_inches="tight")
    print("[umap] saved paper/assets/umap_contrastive.{pdf,png}", flush=True)


if __name__ == "__main__":
    main()
