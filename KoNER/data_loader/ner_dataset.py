import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from .ner_data import NerData
from .utils import *

class NerDataset(Dataset):

    LABEL_BEGIN_TOKEN = "[BEGIN]"
    LABEL_END_TOKEN = "[END]"

    def __init__(
            self,
            vocab_path,
            tokenizer,
            model_type,
            dataset_path=None,
            dataset_fn=None,
            cache_path=None,
            max_len=100):

        super(NerDataset, self).__init__()

        self.vocab_path = vocab_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.token_list = []
        self.label_list = []
        self.PAD_TOKEN = tokenizer.get_pad_token()
        self.PAD_TOKEN_ID = tokenizer.get_pad_token_id()

        self.build(model_type=model_type, dataset_path=dataset_path, dataset_fn=dataset_fn, cache_path=cache_path)

    def build(self, model_type, dataset_path=None, dataset_fn=None, cache_path=None):
        if dataset_path:
            self.dataset_path = os.path.join(dataset_path, dataset_fn)

            cached_data_fn = os.path.join(cache_path, f"{model_type}-{dataset_fn.split('.')[0]}.tmp")
            if os.path.isfile(cached_data_fn):
                data = load_dump(path=cached_data_fn)

                self.token_list = data["tokens"]
                self.label_list = data["labels"]

            else:
                ner_data = NerData(self.dataset_path)

                for data in tqdm(ner_data.data, desc=f"Tokenize sentences for {model_type}"):
                    try:
                        token_list, label_list = self.tokenizer(data["sentence"], data["chars"])

                        self.token_list.append(token_list)
                        self.label_list.append(label_list)
                    except:
                        pass

                save_dump(path=cached_data_fn, data={
                    "tokens": self.token_list,
                    "labels": self.label_list})

        self.labels, self.l2i, self.i2l = self.create_label_dict(self.label_list)

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        tokens, labels, segments = self.get_input_datas(idx)

        token = torch.tensor(tokens)
        label = torch.tensor(labels)
        segments = torch.tensor(segments)

        return token, segments, label

    def create_label_dict(self, label_list=None):
        label_vocab_fn = os.path.join(self.vocab_path, "labels.vocab")
        if os.path.isfile(label_vocab_fn):
            data = load_dump(label_vocab_fn)

            labels = data["labels"]
            l2i = data["l2i"]
            i2l = data["i2l"]

        else:
            # FIXME: Tokenizer와 Label의 PAD TOKEN 값을 맞추기
            labels = [self.LABEL_BEGIN_TOKEN, self.LABEL_END_TOKEN, self.PAD_TOKEN]
            for ll in label_list:
                labels += ll

            labels = list(set(labels))
            l2i = {l: i for i, l in enumerate(labels)}
            i2l = {i: l for i, l in enumerate(labels)}

            save_dump(path=label_vocab_fn, data={
                "labels": labels,
                "l2i": l2i,
                "i2l": i2l
            })

        return labels, l2i, i2l

    def get_input_datas(self, idx):
        tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + self.token_list[idx])
        labels = [self.l2i[self.LABEL_BEGIN_TOKEN]] + [self.l2i[l] for l in self.label_list[idx]]
        segments = [1] * len(tokens)

        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
            labels = labels[:self.max_len]
            segments = segments[:self.max_len]

        elif len(tokens) < self.max_len:
            tokens = tokens + [self.tokenizer.get_pad_token_id()] * (self.max_len - len(tokens))
            labels = labels + [self.l2i[self.PAD_TOKEN]] * (self.max_len - len(labels))
            segments = segments + [0] * (self.max_len - len(segments))

        return tokens, labels, segments