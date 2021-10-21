from dataclasses import dataclass
from . import KoBertTokenizer

@dataclass
class NerTokenizerFactories:
    BERT_TOKENIZER: KoBertTokenizer

class NerTokenizer:
    def __init__(self, tokenizer_name=None):
        self.tokenizer_name = tokenizer_name

        if tokenizer_name == "BERT":
            self.tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
            
    def tokenize(self, sentence):
        return self.tokenizer.tokenize(sentence)

    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_pad_token(self):
        return self.tokenizer.pad_token

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __call__(self, sentence, chars):
        token_list = self.tokenize(sentence)
        label_list = []

        for token in token_list:
            if token.startswith("▁"):
                token = token[1:]

            label = ""
            is_single_token = False
            is_begin_token = False
            is_inside_token = False
            is_end_token = False

            while token:
                token = token[1:]

                while True:
                    char = chars[0]
                    del chars[0]

                    if char["char"] != " ":
                        break

                l = char["label"]

                if l != "O":
                    label = l[2:]

                if l.startswith("S-"):
                    is_single_token = True
                elif l.startswith("B-"):
                    is_begin_token = True
                elif l.startswith("I-"):
                    is_inside_token = True
                elif l.startswith("E-"):
                    is_end_token = True

            if (is_begin_token and is_end_token) or is_single_token:
                label_list.append("S-" + label)
            elif is_end_token:
                label_list.append("E-" + label)
            elif is_begin_token:
                label_list.append("B-" + label)
            elif is_inside_token:
                label_list.append("I-" + label)
            else:
                label_list.append("O")

        return token_list, label_list