from .utils import initialize_directories
from .ko_ner_config import get_ko_ner_configuration
from .tokenizer import NerTokenizer
from .data_loader import NerDataModule
from .models import EmbeddingLayerFactories, MiddleLayerFactories, OutputLayerFactories

from .ner_model_trainer import NerModelTrainer

import torch
import numpy as np
import random

def fix_torch_seed(random_seed=42):
    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)