import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import NerDataset


class NerDataModule(pl.LightningDataModule):
    def __init__(self, config, model_type, tokenizer):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.train_dataset = NerDataset(
            model_type=model_type,
            vocab_path=config.path.vocab,
            tokenizer=tokenizer,
            dataset_path=config.path.data,
            dataset_fn=config.dataset.train,
            cache_path=config.path.cache,
            max_len=config.parameters.max_len)

        self.valid_dataset = NerDataset(
            model_type=model_type,
            vocab_path=config.path.vocab,
            tokenizer=tokenizer,
            dataset_path=config.path.data,
            dataset_fn=config.dataset.valid,
            cache_path=config.path.cache,
            max_len=config.parameters.max_len)

        self.test_dataset = NerDataset(
            model_type=model_type,
            vocab_path=config.path.vocab,
            tokenizer=tokenizer,
            dataset_path=config.path.data,
            dataset_fn=config.dataset.test,
            cache_path=config.path.cache,
            max_len=config.parameters.max_len)

    def get_label_size(self):
        return len(self.train_dataset.labels)

    def get_l2i(self):
        return self.train_dataset.l2i

    def get_i2l(self):
        return self.train_dataset.i2l

    def get_token_pad_id(self):
        return self.tokenizer.get_pad_token_id()

    def get_label_pad_id(self):
        return self.train_dataset.l2i[self.train_dataset.PAD_TOKEN]

    def get_label_start_id(self):
        return self.train_dataset.l2i[self.train_dataset.LABEL_BEGIN_TOKEN]

    def get_label_end_id(self):
        return self.train_dataset.l2i[self.train_dataset.LABEL_END_TOKEN]

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.parameters.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True)