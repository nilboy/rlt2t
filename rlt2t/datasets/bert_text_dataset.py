import json

import torch
from torch.utils.data import Dataset

from rlt2t.textmap.textmap_processor import TextMapProcessor

class BertTextDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 max_token_len: int = 128,
                 start_idx: int = 106,
                 num_words: int = 7000,
                 eos_id: int = 105):
        self.data = []
        with open(file_path) as fin:
            for line in fin:
                self.data.append(json.loads(line))
        self.max_token_len = max_token_len
        self.mapper = TextMapProcessor(start_idx, num_words, eos_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        text_a = self.data[index]['text']
        text_b = self.data[index]['summary']
        label = self.data[index]['label']
        outputs = self.mapper._encode_mlm_single(text_a, text_b, max_length=self.max_token_len,
                                                 pad=True)
        return {
            'input_ids': torch.LongTensor(outputs['input_ids']),
            'token_type_ids': torch.LongTensor(outputs['token_type_ids']),
            'attention_mask': torch.LongTensor(outputs['attention_mask']),
            'labels': torch.tensor([label])
        }
