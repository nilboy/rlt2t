import torch
import json
import random
from torch.utils.data import Dataset

class Rank2Dataset(Dataset):
    def __init__(self,
                 file_path: str,
                 mode: str = 'train'):
        with open(file_path) as fin:
            data = json.load(fin)
        self.records = []
        for k, v in data.items():
            self.records.append(v)
        self.mode = mode
        if self.mode == 'train':
            self.repeat_num = 100
        else:
            self.repeat_num = 1

    def __len__(self):
        return len(self.records) * self.repeat_num

    def __getitem__(self, index: int):
        index = index % len(self.records)
        item1, item2 = random.sample(self.records[index], 2)
        # item1 > item2
        if item1['metric_score'] < item2['metric_score']:
            item1, item2 = item2, item1
        return {
            'x1': torch.FloatTensor(item1['score']),
            'x2': torch.FloatTensor(item2['score'])
        }
