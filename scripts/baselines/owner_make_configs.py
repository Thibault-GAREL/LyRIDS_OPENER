"""Génère les configs TOML OWNER (entity_typing) pour nos datasets exportés.

Une config par dataset, pointant sur les fichiers OWNER produits par
`owner_export.py`. Mode `entity_typing` (typing sur mentions gold + clustering
non-supervisé -> AMI), `save_state = "none"` (train + eval, pas de sauvegarde).

Train source :
  - défaut : le train.json du MÊME dataset (raffinement contrastif in-domain).
  - --train-source <name> : utilise le train.json de <name> pour TOUS (setup
    transfert, ex. conll2003, comme notre embedder contrastif Opener).

Usage :
    python -m scripts.baselines.owner_make_configs
    python -m scripts.baselines.owner_make_configs --train-source conll2003
"""
import argparse
from pathlib import Path

_DEFAULT_DATASETS = [
    'crossner_ai', 'crossner_literature', 'crossner_music',
    'crossner_politics', 'crossner_science',
    'wnut17', 'mit_restaurant', 'mit_movie',
    'fabner', 'bionlp2004', 'conll2003',
    'gum', 'gentle',
]

_ET_BLOCK = '''[entity_typing]
template = "{{sentence}} {{entity}} is a [MASK]."
plm_name = "bert-base-uncased"
max_len = 256
batch_size = 8
num_epochs = 1
learning_rate = 2e-5
k_min = 2
k_max = 30
k_step = 2
'''

_DATA_BLOCK = '''[data]
train_dataset_path = "{train_path}"
train_dataset_name = "{train_name}"
test_dataset_path = "{test_path}"
test_dataset_name = "{ds}"

'''

# typing-sur-gold (clustering sur mentions gold) -> AMI = et_test_ami
_TEMPLATE = 'seed = 100\nmodel = "entity_typing"\nsave_state = "none"\n\n' \
    + _DATA_BLOCK + _ET_BLOCK

# end-to-end (detection DeBERTa + typing) -> AMI = ner_test_ami
_TEMPLATE_NER = 'seed = 100\nmodel = "ner"\nsave_state = "none"\n\n' \
    + _DATA_BLOCK \
    + '[mention_detection]\nplm_name = "microsoft/deberta-v3-base"\n' \
    + 'max_len = 256\nbatch_size = 8\nnum_epochs = 1\nlearning_rate = 2e-5\n\n' \
    + _ET_BLOCK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--data-root', default='external/OWNER/data/lyrids')
    parser.add_argument('--config-dir', default=None,
                        help='Defaut : configs/lyrids (typing) ou configs/lyrids_ner (ner)')
    parser.add_argument('--mode', choices=['typing', 'ner'], default='typing',
                        help='typing = typing-sur-gold ; ner = end-to-end (MD+typing)')
    parser.add_argument('--train-source', default=None,
                        help='Si fourni, train.json de ce dataset pour TOUS (transfert).')
    args = parser.parse_args()

    template = _TEMPLATE if args.mode == 'typing' else _TEMPLATE_NER
    cfg_dir = Path(args.config_dir or
                   f'external/OWNER/configs/lyrids{"" if args.mode == "typing" else "_ner"}')
    data_root = Path(args.data_root).resolve()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or _DEFAULT_DATASETS

    written = []
    for ds in datasets:
        src = args.train_source or ds
        train_path = (data_root / src / 'train.json').as_posix()
        test_path = (data_root / ds / 'test.json').as_posix()
        toml = template.format(train_path=train_path, train_name=src,
                               test_path=test_path, ds=ds)
        out = cfg_dir / f'{ds}.toml'
        out.write_text(toml, encoding='utf-8')
        written.append(out)
        print(f"  {ds:<22} train_source={src:<12} -> {out}")

    print(f"\n{len(written)} configs écrites dans {cfg_dir}")
    print("Lancer chaque run (dans l'env conda OWNER) :")
    print("  mlflow run -e ner -P config_file=<config.toml> external/OWNER")


if __name__ == '__main__':
    main()
