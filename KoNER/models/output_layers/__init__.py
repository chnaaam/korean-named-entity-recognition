from dataclasses import dataclass
from .crf import CRFLayer

from .output_layer import OutputLayer

@dataclass
class OutputLayerFactories:
    CRF = CRFLayer

    get_layer = {
        "CRF": CRF
    }

__all__ = [
    "OutputLayerFactories",
    "OutputLayer"
]