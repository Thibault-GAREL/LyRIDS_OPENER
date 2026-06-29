"""OPENER: turnkey open-world Named Entity Recognition.

    from opener import OpenerZS, OpenerSup

    # zero-shot: no labels to train, just give the candidate type names
    m = OpenerZS.from_pretrained("Thibault-GAREL/opener-zs")
    m.predict("Marie Curie discovered radium.", labels=["person", "discovery"])

    # supervised: fit a tiny head on your labelled data, then predict
    m = OpenerSup.from_pretrained("Thibault-GAREL/opener-sup")
    m.fit(train_texts, train_annotations).predict("…")
"""
from .zs import OpenerZS
from .sup import OpenerSup

__all__ = ["OpenerZS", "OpenerSup"]
__version__ = "0.1.0"
