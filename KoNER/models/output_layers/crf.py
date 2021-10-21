import torch.nn as nn
from torchcrf import CRF


class CRFLayer(nn.Module):
    def __init__(self, parameters):
        super(CRFLayer, self).__init__()

        self.build_layer(parameters)

    def __str__(self):
        return "CRF"

    def build_layer(self, parameters):
        if "label_size" not in parameters:
            raise KeyError()

        self.fc = nn.Linear(
            in_features=parameters["fc_in_size"],
            out_features=parameters["label_size"])

        self.crf = CRF(num_tags=parameters["label_size"], batch_first=True)

    def forward(self, features, parameters):
        emissions = self.fc(features)
        labels = parameters["labels"]
        crf_masks = parameters["crf_masks"]

        if labels is not None:
            log_likelihood = self.crf(emissions, labels, crf_masks)
            sequence_of_tags = self.crf.decode(emissions, crf_masks)

            return (-1) * log_likelihood, sequence_of_tags
        else:
            sequence_of_tags = self.crf.decode(emissions, crf_masks)

            return sequence_of_tags
