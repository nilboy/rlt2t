from transformers import BatchEncoding
from typing import List

class TextMapProcessor(object):
    def __init__(self, start_idx: int = 106,
                 num_words: int = 7000,
                 eos_id: int = 105,
                 bos_id: int = 104,
                 cls_token_id: int = 101,
                 sep_token_id: int = 102):
        self.start_idx = start_idx
        self.num_words = num_words
        self.eos_id = eos_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.bos_id = bos_id

    def decode(self, tids: List[int]):
        return " ".join([str(tid - self.start_idx) for tid in tids])

    def encode_t5(self, inputs: List[str],
                  max_length, add_special_tokens=True,
                  pad=False):
        outputs = {
            'input_ids': [],
            'attention_mask': []
        }
        for text in inputs:
            tids = [int(token) + self.start_idx for token in text.split()]
            if add_special_tokens:
                tids = tids[0:max_length-1]
                tids.append(self.eos_id)
            else:
                tids = tids[0:max_length]
            attention_mask = [1] * len(tids)
            if pad:
                self.pad_to_max_length(tids, max_length)
                self.pad_to_max_length(attention_mask, max_length)
            outputs['input_ids'].append(tids)
            outputs['attention_mask'].append(attention_mask)
        return BatchEncoding(data=outputs)

    def encode_gpt2(self, inputs: List[str],
                    max_length):
        outputs = {
            'input_ids': [],
            'attention_mask': []
        }
        for text in inputs:
            tids = [self.bos_id]
            tids.extend([int(token) + self.start_idx for token in text.split()])
            if len(tids) > max_length - 1:
                tids = tids[0:max_length-1]
            tids.append(self.eos_id)
            attention_mask = [1] * len(tids)
            outputs['input_ids'].append(tids)
            outputs['attention_mask'].append(attention_mask)
        return BatchEncoding(data=outputs)

    def encode_mlm(self, texts_a: List[str], texts_b: List[str], max_length: int):
        outputs = {
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'special_tokens_mask': []
        }
        for text_a, text_b in zip(texts_a, texts_b):
            item = self._encode_mlm_single(text_a, text_b, max_length)
            for k, v in item.items():
                outputs[k].append(v)
        return BatchEncoding(data=outputs)

    def _encode_mlm_single(self, text_a: str, text_b: str, max_length: int,
                           pad=False):
        text_a_tids = [int(token) + self.start_idx for token in text_a.split()]
        text_b_tids = [int(token) + self.start_idx for token in text_b.split()]
        if len(text_a_tids) > max_length - 3:
            text_a_tids = text_a_tids[0:max_length - 3]
        if len(text_b_tids) > max_length - 3 - len(text_a_tids):
            text_b_tids = text_b_tids[0:max_length - 3 - len(text_a_tids)]
        input_ids, token_type_ids, attention_mask, special_tokens_mask = [], [], [], []
        # [CLS]
        input_ids.append(self.cls_token_id)
        token_type_ids.append(0)
        attention_mask.append(1)
        special_tokens_mask.append(1)
        # text_a
        input_ids.extend(text_a_tids)
        token_type_ids.extend([0] * len(text_a_tids))
        attention_mask.extend([1] * len(text_a_tids))
        special_tokens_mask.extend([0] * len(text_a_tids))
        # [SEP]
        input_ids.append(self.sep_token_id)
        token_type_ids.append(0)
        attention_mask.append(1)
        special_tokens_mask.append(1)
        # text_b
        input_ids.extend(text_b_tids)
        token_type_ids.extend([1] * len(text_b_tids))
        attention_mask.extend([1] * len(text_b_tids))
        special_tokens_mask.extend([0] * len(text_b_tids))
        # [SEP]
        input_ids.append(self.sep_token_id)
        token_type_ids.append(1)
        attention_mask.append(1)
        special_tokens_mask.append(1)
        if pad:
            self.pad_to_max_length(input_ids, max_length)
            self.pad_to_max_length(token_type_ids, max_length)
            self.pad_to_max_length(attention_mask, max_length)
            self.pad_to_max_length(special_tokens_mask, max_length)
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'special_tokens_mask': special_tokens_mask
        }

    def pad_to_max_length(self, values, max_length):
        if len(values) < max_length:
            pad_length = max_length - len(values)
            values.extend([0] * pad_length)