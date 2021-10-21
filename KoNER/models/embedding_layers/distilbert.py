import torch.nn as nn

from transformers import BertModel

MODEL_NAME = "monologg/distilkobert"

class DistilBertEmbeddingLayer(nn.Module):
    def __init__(self, **parameters):
        super(DistilBertEmbeddingLayer, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME)

    def __str__(self):
        return "distilbert"

    # def forward(self, *args, **kwargs):
    #     # x, token_type_ids, attention_mask
    #     return self.bert(x, token_type_ids=token_type_ids, attention_mask=attention_mask)
