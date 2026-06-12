"""Genere les configs OWNER en protocole TRANSFERT (= protocole du papier OWNER).

OWNER (papier) entraine son modele UNE fois sur une source (conll2003), puis le
CHARGE (load_finetuned) pour tester chaque cible en zero-shot (cf.
external/OWNER/configs/reproducibility/runs.csv : 1er run save_finetuned, le
reste load_finetuned). C'est different de l'in-domain (un entrainement par
dataset) qu'on avait fait par erreur.

Genere, dans configs/lyrids_ner_transfer/ :
  - <source>__train.toml : train=conll2003, test=conll2003, model=ner,
    save_finetuned -> entraine MD+typing sur conll2003 et SAUVE (save_path).
  - <ds>.toml (x13)      : train=conll2003 (ignore), test=<ds>, load_finetuned
    -> CHARGE le modele sauve et teste la cible (inference: detecter+encoder+cluster).

A lancer dans cet ordre (le train d'abord) -> cf. run_owner_transfer.ps1.

Usage : python -m scripts.baselines.owner_make_transfer --source conll2003 --epochs 4
"""
import argparse
from pathlib import Path

_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003', 'gum', 'gentle',
]


def _cfg(model_extra, train_path, test_path, ds, save_path, save_state, epochs):
    return f'''seed = 100
model = "ner"
save_state = "{save_state}"
save_path = "{save_path}"

[data]
train_dataset_path = "{train_path}"
train_dataset_name = "{train_path.split('/')[-2]}"
test_dataset_path = "{test_path}"
test_dataset_name = "{ds}"

[mention_detection]
plm_name = "microsoft/deberta-v3-base"
max_len = 256
batch_size = 4
num_epochs = {epochs}
learning_rate = 2e-5

[entity_typing]
template = "{{sentence}} {{entity}} is a [MASK]."
plm_name = "bert-base-uncased"
max_len = 256
batch_size = 4
num_epochs = {epochs}
learning_rate = 2e-5
k_min = 2
k_max = 30
k_step = 2
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='conll2003')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Epochs pour l ENTRAINEMENT source unique (les load ne s entrainent pas).')
    parser.add_argument('--data-root', default='external/OWNER/data/lyrids')
    parser.add_argument('--config-dir', default='external/OWNER/configs/lyrids_ner_transfer')
    parser.add_argument('--save-path', default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    cfg_dir = Path(args.config_dir); cfg_dir.mkdir(parents=True, exist_ok=True)
    save_path = (Path(args.save_path).resolve().as_posix() if args.save_path
                 else (Path('external/OWNER/checkpoints').resolve()
                       / f'transfer_{args.source}' / '100').as_posix())
    src_train = (data_root / args.source / 'train.json').as_posix()

    # 1) config d'entrainement source (save_finetuned) : train+test conll2003, sauve
    src_test = (data_root / args.source / 'test.json').as_posix()
    (cfg_dir / f'{args.source}__train.toml').write_text(
        _cfg(None, src_train, src_test, args.source, save_path,
             'save_finetuned', args.epochs), encoding='utf-8')

    # 2) configs de test (load_finetuned) pour chaque cible
    for ds in _DEFAULT_DATASETS:
        test_path = (data_root / ds / 'test.json').as_posix()
        (cfg_dir / f'{ds}.toml').write_text(
            _cfg(None, src_train, test_path, ds, save_path,
                 'load_finetuned', args.epochs), encoding='utf-8')

    print(f"Configs transfert ecrites dans {cfg_dir}")
    print(f"  source train+save : {args.source}__train.toml (save_finetuned, {args.epochs} ep)")
    print(f"  cibles (load)     : {len(_DEFAULT_DATASETS)} configs load_finetuned")
    print(f"  save_path         : {save_path}")
    print(f"\n⚠️ conll2003 cible = IN-DOMAIN (= source) -> a noter dans le papier.")


if __name__ == '__main__':
    main()
