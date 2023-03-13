from transformers import PreTrainedTokenizer
from typing import List


class TextMapProcessor(object):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 start_idx: int = 3,
                 num_words: int = 5000):
        self.start_idx = start_idx
        tokens = tokenizer.convert_ids_tokens(range(start_idx, start_idx + num_words))
        ids = range(start_idx, start_idx + num_words)
        self.token2id = {
            token: id
            for token, id in zip(tokens, ids)
        }
        self.id2token = {
            id: token
            for token, id in zip(tokens, ids)
        }

    def tokenize(self, text):
        token_ids = [
            int(token)
            for token in text.split()
        ]
        return [
            self.id2token[tid]
            for tid in token_ids
        ]

    def decode(self, tids: List[int]):
        return " ".join([str(tid - self.start_idx) for tid in tids])
