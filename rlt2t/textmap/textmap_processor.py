from transformers import BatchEncoding
from typing import List

class TextMapProcessor(object):
    def __init__(self, start_idx: int = 106,
                 num_words: int = 7000,
                 eos_id: int = 105):
        self.start_idx = start_idx
        self.num_words = num_words
        self.eos_id = eos_id

    def decode(self, tids: List[int]):
        return " ".join([str(tid - self.start_idx) for tid in tids])

    def encode_t5(self, inputs: List[str],
                  max_length, add_special_tokens=True):
        outputs = {
            'input_ids': [],
            'attention_mask': []
        }
        for text in inputs:
            tids = [int(token) + self.start_idx for token in text.split()]
            tids = tids[0:max_length]
            if add_special_tokens:
                tids.append(self.eos_id)
            attention_mask = [1] * len(tids)
            outputs['input_ids'].append(tids)
            outputs['attention_mask'].append(attention_mask)
        return BatchEncoding(data=outputs)
