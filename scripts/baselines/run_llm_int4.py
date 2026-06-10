"""Baselines LLM 7B en int4 (bitsandbytes NF4) pour le benchmark open-world NER.

Wrapper GÉNÉRIQUE à adaptateurs : un même runner instrumenté (AMI / vitesse /
énergie + sauvegarde incrémentale + --resume), et un adaptateur par modèle qui
définit (1) le checkpoint, (2) la construction du prompt, (3) le parsing de la
sortie vers des spans (start, end, label).

Modèles supportés (tous 7B -> ~4 Go en int4, tiennent sur 6 Go VRAM) :

  --model gner_llama   dyyyyyyyy/GNER-LLaMA-7B
        Génératif BIO, format `word(tag)` (même parser que run_gner.py).
  --model uniner       Universal-NER/UniNER-7B-all
        Conversationnel : une requête PAR TYPE d'entité, sortie = liste JSON de
        mentions, re-mappées sur des offsets caractères.
  --model uniner_def   Universal-NER/UniNER-7B-definition
        Idem UniNER (variante entraînée avec définitions).
  --model gollie       HiTZ/GoLLIE-7B   [EXPÉRIMENTAL]
        GoLLIE attend un prompt-code Python très spécifique (scaffolding du repo
        HiTZ/GoLLIE). Ici on utilise un adaptateur instruct générique (sortie
        JSON [{"text","type"}]) -> chiffres NON officiels, à valider.

Protocole d'éval identique à run_gliner.py / run_gner.py : alignement gold/pred
par offsets exacts (Eq. 7 OWNER) + sentinels FN/FP -> AMI / Macro-F1 / accuracy.

⚠️ COÛT : ces modèles sont 7B, génératifs, en int4 -> LENTS sur GTX 1660 Ti.
UniNER interroge le modèle UNE FOIS PAR TYPE et par phrase (coût ~ K_labels ×
N_phrases générations). `--max-eval` est volontairement modeste par défaut.
Lancer en process détaché (cf. scripts/run_baselines_queue.ps1).
"""
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             f1_score)

from src.data.owner_datasets import load_owner_dataset
from src.utils.energy import measure_energy
from src.utils.timing import LatencyMeter
# Réutilise le parser GNER (word(tag) -> spans) pour l'adaptateur GNER-LLaMA.
from scripts.baselines.run_gner import (PROMPT_TEMPLATE, _decode_to_spans,
                                        collect_label_set)


# Sentinels FP/FN (cf. OWNER paper Section IV.C, identiques aux autres baselines)
LABEL_FN = '__gold_not_predicted__'
LABEL_FP = '__predicted_not_gold__'


def align(gold_spans, pred_spans):
    gold_d = {(s, e): lbl for s, e, lbl in gold_spans}
    pred_d = {(s, e): lbl for s, e, lbl in pred_spans}
    all_keys = set(gold_d) | set(pred_d)
    y_gold, y_pred = [], []
    for k in all_keys:
        y_gold.append(gold_d.get(k, LABEL_FP))
        y_pred.append(pred_d.get(k, LABEL_FN))
    return y_gold, y_pred


# ---------- Génération causale (decoder-only) ----------

def _gen_causal(model, tokenizer, prompt, max_new_tokens, max_input_length=2048):
    import torch
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                       max_length=max_input_length).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    new_ids = out[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------- Utilitaires de mapping mention -> offsets ----------

def _find_span(text, mention, used):
    """Première occurrence exacte de `mention` dans `text` non chevauchante
    avec `used` (liste de (s, e)). Retourne (s, e) ou None."""
    mention = mention.strip()
    if not mention:
        return None
    start = 0
    while True:
        i = text.find(mention, start)
        if i < 0:
            return None
        j = i + len(mention)
        if all(not (i < ue and us < j) for (us, ue) in used):
            return (i, j)
        start = i + 1


def _parse_json_string_list(out):
    """Extrait une liste de chaînes depuis une sortie type '["a", "b"]'."""
    m = re.search(r'\[.*?\]', out, re.S)
    if not m:
        return []
    blob = m.group(0)
    try:
        arr = json.loads(blob)
        items = [a for a in arr if isinstance(a, str)]
    except Exception:
        items = re.findall(r'"([^"]*)"', blob)
    return [a.strip() for a in items if a and a.strip()]


# ======================= ADAPTATEURS =======================

class GnerLlamaAdapter:
    key = 'gner_llama'
    checkpoint = 'dyyyyyyyy/GNER-LLaMA-7B'
    model_kind = 'causal'

    def predict(self, model, tokenizer, text, labels, max_new_tokens):
        instruction = PROMPT_TEMPLATE.format(labels=', '.join(labels), sentence=text)
        prompt = f"[INST] {instruction} [/INST]"
        out = _gen_causal(model, tokenizer, prompt, max_new_tokens)
        return _decode_to_spans(text, out)


class UniNERAdapter:
    key = 'uniner'
    checkpoint = 'Universal-NER/UniNER-7B-all'
    model_kind = 'causal'

    @staticmethod
    def _conv(text, entity_type):
        return (
            "A virtual assistant answers questions from a user based on the "
            "provided text.\n"
            f"USER: Text: {text}\n"
            "ASSISTANT: I've read this text.\n"
            f"USER: What describes {entity_type} in the text?\n"
            "ASSISTANT:")

    def predict(self, model, tokenizer, text, labels, max_new_tokens):
        spans, used = [], []
        for lbl in labels:
            query = lbl.replace('_', ' ')
            out = _gen_causal(model, tokenizer, self._conv(text, query),
                              max_new_tokens=min(max_new_tokens, 256))
            for mention in _parse_json_string_list(out):
                sp = _find_span(text, mention, used)
                if sp:
                    used.append(sp)
                    spans.append((sp[0], sp[1], lbl))
        return spans


class UniNERDefAdapter(UniNERAdapter):
    key = 'uniner_def'
    checkpoint = 'Universal-NER/UniNER-7B-definition'


class GollieAdapter:
    """EXPÉRIMENTAL : adaptateur instruct générique (PAS le format GoLLIE officiel).

    Le vrai GoLLIE exige un prompt-code Python (dataclasses + guidelines) issu
    du repo HiTZ/GoLLIE. Ici on demande une extraction JSON simple -> à valider /
    remplacer par le scaffolding officiel pour des chiffres publiables.
    """
    key = 'gollie'
    checkpoint = 'HiTZ/GoLLIE-7B'
    model_kind = 'causal'

    def predict(self, model, tokenizer, text, labels, max_new_tokens):
        types = ', '.join(labels)
        prompt = (
            "Extract named entities from the text. Allowed types: "
            f"{types}.\n"
            'Answer ONLY with a JSON list of objects {"text": <span>, '
            '"type": <one allowed type>}.\n'
            f"Text: {text}\nJSON:")
        out = _gen_causal(model, tokenizer, prompt, max_new_tokens)
        spans, used = [], []
        m = re.search(r'\[.*\]', out, re.S)
        if not m:
            return spans
        try:
            arr = json.loads(m.group(0))
        except Exception:
            return spans
        labelset = set(labels)
        for obj in arr:
            if not isinstance(obj, dict):
                continue
            mention, typ = obj.get('text'), obj.get('type')
            if not mention or typ not in labelset:
                continue
            sp = _find_span(text, str(mention), used)
            if sp:
                used.append(sp)
                spans.append((sp[0], sp[1], typ))
        return spans


ADAPTERS = {a.key: a for a in [
    GnerLlamaAdapter(), UniNERAdapter(), UniNERDefAdapter(), GollieAdapter()]}


# ======================= RUNNER =======================

def run_dataset(adapter, model, tokenizer, name, max_eval, max_new_tokens):
    print(f"=== {name} ===")
    test = load_owner_dataset(name, split='test', max_sentences=max_eval)
    if not test:
        return {'status': 'empty_test_split'}
    labels = collect_label_set(test)
    print(f"  {len(test)} sentences, {len(labels)} labels")

    meter = LatencyMeter()
    meter.warmup(
        lambda: adapter.predict(model, tokenizer, 'hello world', labels, max_new_tokens),
        n=1)

    all_y_gold, all_y_pred = [], []
    n_gold = n_pred = n_matched = 0
    with measure_energy(project=f'{adapter.key}-{name}') as energy:
        for text, gold_spans in test:
            with meter.measure():
                pred_spans = adapter.predict(model, tokenizer, text, labels, max_new_tokens)
            y_g, y_p = align(gold_spans, pred_spans)
            all_y_gold.extend(y_g)
            all_y_pred.extend(y_p)
            n_gold += len(gold_spans)
            n_pred += len(pred_spans)
            n_matched += len(
                {(s, e) for s, e, _ in gold_spans}
                & {(s, e) for s, e, _ in pred_spans})

    ami = float(adjusted_mutual_info_score(all_y_gold, all_y_pred))
    acc = float(accuracy_score(all_y_gold, all_y_pred))
    f1m = float(f1_score(all_y_gold, all_y_pred, average='macro', zero_division=0))
    energy_rep = energy.report.as_dict()
    timing = meter.stats().as_dict()

    print(f"  AMI={ami:.4f}  acc={acc:.4f}  macro_f1={f1m:.4f}  "
          f"matched={n_matched}/{n_gold} gold  ({n_pred} pred)")
    print(f"  energy : {energy_rep}")
    print(f"  timing : p50={timing['p50_ms']:.1f}ms  thrpt={timing['throughput_per_s']:.2f} sent/s")

    return {
        'n_eval_sentences': len(test),
        'n_labels': len(labels),
        'n_gold_spans': n_gold,
        'n_pred_spans': n_pred,
        'n_matched_offsets': n_matched,
        'ami': round(ami, 4),
        'accuracy': round(acc, 4),
        'macro_f1': round(f1m, 4),
        'energy': energy_rep,
        'timing_inference': timing,
    }


_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003',
    'gum', 'gentle',
]


def purge_model_cache(checkpoint):
    """Supprime les poids du modèle du cache HF (API officielle huggingface_hub).

    Évite l'accumulation disque (chaque 7B ~13 Go). Appelé en fin de run si
    --purge-cache. N'utilise PAS Remove-Item (bloqué par hook PowerShell)."""
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        hashes = [r.commit_hash for repo in cache.repos
                  if repo.repo_id == checkpoint for r in repo.revisions]
        if hashes:
            freed = cache.delete_revisions(*hashes).execute()
            print(f"[purge] {checkpoint} supprimé du cache "
                  f"({len(hashes)} révision(s)).")
        else:
            print(f"[purge] rien à supprimer pour {checkpoint}.")
    except Exception as e:
        print(f"[purge] échec ({e!r}) — à nettoyer manuellement si besoin.")


def load_int4_model(checkpoint):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Anti-pic-RAM au chargement (machine 16 Go) : low_cpu_mem_usage charge
    # shard par shard, offload_state_dict ecrit le reste sur disque au lieu de
    # saturer la RAM. Le modele int4 final (~4 Go) tient sur le GPU 6 Go.
    offload_dir = os.path.join(os.environ.get('HF_HOME', '.'), 'offload')
    os.makedirs(offload_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, quantization_config=bnb, device_map='auto',
        trust_remote_code=True, low_cpu_mem_usage=True,
        offload_state_dict=True, offload_folder=offload_dir)
    model.eval()
    return model, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(ADAPTERS),
                        help='Adaptateur / modèle 7B int4 à évaluer.')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--max-eval', type=int, default=200,
                        help='Phrases test par dataset (modeste car 7B génératif lent).')
    parser.add_argument('--max-new-tokens', type=int, default=640)
    parser.add_argument('--output-dir', default=None,
                        help='Défaut : outputs/results/baselines/<model>')
    parser.add_argument('--resume', action='store_true',
                        help='Saute les datasets déjà présents (status ok).')
    parser.add_argument('--purge-cache', action='store_true',
                        help='Supprime les poids du modèle du cache HF en fin de '
                             'run (évite ~13 Go/modèle qui s accumulent).')
    parser.add_argument('--hf-cache', default=None,
                        help='Répertoire cache HF pour CE run (ex: D:\\hf_cache). '
                             'Redirige modèles + datasets si C: est plein.')
    args = parser.parse_args()

    # Downloader classique (robuste) plutot que hf_xet : ce dernier peut crasher
    # SILENCIEUSEMENT (sortie sans traceback) sur un partiel corrompu ou un
    # reseau instable. Le classique donne aussi de vraies barres de progression.
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

    # IMPORTANT : doit être positionné AVANT tout import de transformers /
    # huggingface_hub (faits paresseusement dans load_int4_model).
    if args.hf_cache:
        os.environ['HF_HOME'] = args.hf_cache
        os.environ['HF_HUB_CACHE'] = os.path.join(args.hf_cache, 'hub')
        os.makedirs(os.environ['HF_HUB_CACHE'], exist_ok=True)
        print(f"[hf-cache] HF_HOME -> {args.hf_cache}")

    adapter = ADAPTERS[args.model]
    out_dir = Path(args.output_dir or f'outputs/results/baselines/{adapter.key}')
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_json = out_dir / f'{adapter.key}_progress.json'

    if adapter.key == 'gollie':
        print("⚠️  GoLLIE = adaptateur instruct générique EXPÉRIMENTAL "
              "(pas le format-code officiel). Chiffres à valider.")

    results = {}
    if args.resume and progress_json.exists():
        try:
            results = json.loads(progress_json.read_text(encoding='utf-8')).get('results', {})
            done = [k for k, v in results.items() if 'ami' in v]
            print(f"[resume] {len(done)} datasets déjà faits : {done}")
        except Exception as e:
            print(f"[resume] lecture impossible ({e!r}), on repart de zéro.")

    def _flush():
        progress_json.write_text(json.dumps(
            {'params': vars(args), 'results': results},
            indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"Chargement {adapter.checkpoint} en int4 (NF4)...")
    model, tokenizer = load_int4_model(adapter.checkpoint)
    print(f"Chargé sur {model.device}")

    datasets = args.datasets or _DEFAULT_DATASETS
    print(f"Datasets : {datasets}  (max_eval={args.max_eval})")

    for name in datasets:
        if args.resume and name in results and 'ami' in results[name]:
            print(f"=== {name} === [skip: déjà fait]")
            continue
        try:
            results[name] = run_dataset(adapter, model, tokenizer, name,
                                        args.max_eval, args.max_new_tokens)
        except Exception as e:
            import traceback
            print(f"  CRASH on {name}: {e!r}")
            traceback.print_exc()
            results[name] = {'status': 'crashed', 'error': repr(e)[:300]}
        _flush()

    date_str = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    out_json = out_dir / f'{adapter.key}_{date_str}.json'
    out_json.write_text(json.dumps({'params': vars(args), 'results': results},
                                   indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nJSON : {out_json}")
    print(f"Progress : {progress_json}")

    if args.purge_cache:
        try:
            del model
        except Exception:
            pass
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        purge_model_cache(adapter.checkpoint)


if __name__ == '__main__':
    main()
