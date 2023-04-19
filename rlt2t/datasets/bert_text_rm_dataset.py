import json
import random
import torch
from torch.utils.data import Dataset

from rlt2t.textmap.textmap_processor import TextMapProcessor

class BertTextRMDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 max_token_len: int = 128,
                 start_idx: int = 106,
                 num_words: int = 7000,
                 eos_id: int = 105, mode='train'):
        self.data = []
        with open(file_path) as fin:
            for line in fin:
                self.data.append(json.loads(line))
        self.max_token_len = max_token_len
        self.mapper = TextMapProcessor(start_idx, num_words, eos_id)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        text = self.data[index]['text']
        summary_a = self.data[index]['summary_a']
        summary_b = self.data[index]['summary_b']

        outputs_a = self.mapper._encode_mlm_single(text, summary_a, max_length=self.max_token_len,
                                                   pad=True)
        outputs_b = self.mapper._encode_mlm_single(text, summary_b, max_length=self.max_token_len,
                                                   pad=True)
        return {
            'input_ids_a': torch.LongTensor(outputs_a['input_ids']),
            'token_type_ids_a': torch.LongTensor(outputs_a['token_type_ids']),
            'attention_mask_a': torch.LongTensor(outputs_a['attention_mask']),
            'input_ids_b': torch.LongTensor(outputs_b['input_ids']),
            'token_type_ids_b': torch.LongTensor(outputs_b['token_type_ids']),
            'attention_mask_b': torch.LongTensor(outputs_b['attention_mask']),
            'score_a': torch.tensor(self.data[index]['score_a']),
            'score_b': torch.tensor(self.data[index]['score_b'])
        }

    # def __getitem__(self, index: int):
    #     text = self.data[index]['text']
    #     if self.mode == 'train':
    #         item_a, item_b = random.sample(self.data[index]['summary_list'], 2)
    #     else:
    #         item_a, item_b = self.data[index]['summary_list'][-2], self.data[index]['summary_list'][-1]
    #     if item_a['score'] < item_b['score']:
    #         item_a, item_b = item_b, item_a
    #
    #     summary_a = item_a['summary']
    #     summary_b = item_b['summary']
    #
    #     outputs_a = self.mapper._encode_mlm_single(text, summary_a, max_length=self.max_token_len,
    #                                                pad=True)
    #     outputs_b = self.mapper._encode_mlm_single(text, summary_b, max_length=self.max_token_len,
    #                                                pad=True)
    #     return {
    #         'input_ids_a': torch.LongTensor(outputs_a['input_ids']),
    #         'token_type_ids_a': torch.LongTensor(outputs_a['token_type_ids']),
    #         'attention_mask_a': torch.LongTensor(outputs_a['attention_mask']),
    #         'input_ids_b': torch.LongTensor(outputs_b['input_ids']),
    #         'token_type_ids_b': torch.LongTensor(outputs_b['token_type_ids']),
    #         'attention_mask_b': torch.LongTensor(outputs_b['attention_mask']),
    #         'score_a': torch.tensor(item_a['score']),
    #         'score_b': torch.tensor(item_b['score'])
    #     }
