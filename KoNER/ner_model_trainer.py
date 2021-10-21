import pytorch_lightning as pl

import torch.optim as optim

from .models import NerModel, EmbeddingLayerFactories, MiddleLayerFactories, OutputLayerFactories
from .metrics import calc_f1_score


class NerModelTrainer(pl.LightningModule):
    def __init__(self, config, l2i, i2l, token_pad_id, label_pad_id, label_start_id, label_end_id):
        super().__init__()

        self.config = config
        self.l2i = l2i
        self.i2l = i2l
        self.token_pad_id = token_pad_id
        self.label_pad_id = label_pad_id
        self.label_start_id = label_start_id
        self.label_end_id = label_end_id

        self.model = NerModel(
            embedding_layer=EmbeddingLayerFactories.get_layer[self.config.model.embedding_layer],
            middle_layer=MiddleLayerFactories.get_layer[self.config.model.middle_layer],
            output_layer=OutputLayerFactories.get_layer[self.config.model.output_layer],

            fc_in_size=self.config.parameters.fc_in_size,
            label_size=len(l2i))

    def forward(self, tokens, **parameters):
        return self.model(tokens=tokens, parameters=parameters)

    def training_step(self, batch, batch_idx):
        tokens, token_type_ids, labels = batch

        loss, pred_tags = self(
            tokens=tokens,
            token_type_ids=token_type_ids,
            attention_mask=(tokens != 1).float(),
            labels=labels,
            crf_masks=(tokens != self.token_pad_id))

        true_y, pred_y = self.decode(labels=labels, pred_tags=pred_tags)

        score = calc_f1_score(true_y, pred_y)

        self.log("train_loss", loss)
        self.log("train_f1_score", score * 100)

        return loss

    def validation_step(self, batch, batch_idx):
        tokens, token_type_ids, labels = batch

        loss, pred_tags = self(
            tokens=tokens,
            token_type_ids=token_type_ids,
            attention_mask=(tokens != self.token_pad_id),
            labels=labels,
            crf_masks=(tokens != self.token_pad_id))

        true_y, pred_y = self.decode(labels=labels, pred_tags=pred_tags)

        score = calc_f1_score(true_y, pred_y)
        print(score)

        self.log("val_loss", loss)
        self.log("val_f1_score", score * 100)

    # region Test Step
    # def test_step(self, batch, batch_idx):
    #     tokens, token_type_ids, labels = batch
    #
    #     attention_mask = (tokens != self.pad_id).float()
    #
    #     pred_tags = self(
    #         x=tokens,
    #         token_type_ids=token_type_ids,
    #         attention_mask=attention_mask,
    #         labels=None,
    #         mask=None)
    #
    #     y_true = []
    #     y_pred = []
    #
    #     for idx, label in enumerate(labels):
    #         true = []
    #         pred = []
    #         for jdx in range(len(label)):
    #             if label[jdx] == self.pad_id:
    #                 break
    #
    #             if pred_tags[idx][jdx] == self.pad_id:
    #                 pred_tags[idx][jdx] = self.l2i["O"]
    #
    #             true.append(self.i2l[label[jdx].item()])
    #             pred.append(self.i2l[pred_tags[idx][jdx]])
    #
    #         y_true.append(true)
    #         y_pred.append(pred)
    #
    #     # for i in range(len(y_pred)):
    #     #     for j in range(1, len(y_pred[i])):
    #     #         if y_pred[i][j-1] == "O" and "E-" in y_pred[i][j]:
    #     #             y_pred[i][j] = y_pred[i][j].replace('E-', 'S-')
    #
    #                 # y_pred[i][j-1] = y_pred[i][j].replace('E-', 'B-')
    #
    #     origin_tokens = []
    #     for i in range(len(tokens)):
    #         origin_tokens.append([])
    #
    #         for j in range(len(tokens[i])):
    #             origin_tokens[i].append(self.i2w[tokens[i][j].item()])
    #
    #     # O, B-PS, I-PS, E-PS, O, E-CV, O, O, O, O, S-CV, O, O, O, O, O, O, O
    #     with open("test.txt", "a", encoding="utf-8") as fp:
    #
    #         for i in range(len(y_pred)):
    #             fp.write("S : " + ", ".join(origin_tokens[i][:len(y_pred[i])]) + "\n")
    #             fp.write("T : " + ", ".join(y_true[i][:len(y_pred[i])]) + "\n")
    #             fp.write("P : " + ", ".join(y_pred[i]) + "\n")
    #             fp.write("\n")
    #
    #     score = f1_score(y_true, y_pred, mode="strict", scheme=IOBES)
    #     return {"f1_score": score * 100}
    #
    # def test_epoch_end(self, outputs):
    #     avg_f1_score = torch.tensor([x['f1_score'] for x in outputs]).mean()
    #
    #     self.log("f1_score", avg_f1_score)
    #endregion

    def configure_optimizers(self):
        # param_optimizer = list(self.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
        #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optim.AdamW(
            self.parameters(),
            lr=float(self.config.parameters.learning_rate),
            weight_decay=self.config.parameters.weight_decay)

    def decode(self, labels, pred_tags):
        true_y = []
        pred_y = []

        for idx, label in enumerate(labels):
            true = []
            pred = []

            for jdx in range(len(label)):

                if label[jdx] == self.label_pad_id:
                    break

                if label[jdx] == self.label_start_id:
                    continue

                if pred_tags[idx][jdx] == self.label_pad_id or pred_tags[idx][jdx] == self.label_start_id or \
                        pred_tags[idx][jdx] == self.label_end_id:
                    pred_tags[idx][jdx] = self.l2i["O"]

                true.append(self.i2l[label[jdx].item()])
                pred.append(self.i2l[pred_tags[idx][jdx]])

            true_y.append(true)
            pred_y.append(pred)

        return true_y, pred_y