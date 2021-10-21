from dataclasses import dataclass

from .bert import BertEmbeddingLayer
from .embedding_layer import EmbeddingLayer

@dataclass
class EmbeddingLayerFactories:
    BERT = BertEmbeddingLayer

    get_layer = {
        "BERT": BERT,
    }


__all__ = [
    "EmbeddingLayerFactories", 
    "EmbeddingLayer"
]