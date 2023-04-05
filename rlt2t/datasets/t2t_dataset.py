import json
import random

import torch
from torch.utils.data import Dataset

from rlt2t.textmap.textmap_processor import TextMapProcessor

def augment_text_fn(my_str):
    # 将字符串拆分成单词列表
    words = my_str.split()
    # 计算需要删除或重复的单词数量
    num_to_modify = int(len(words) * 0.15)
    # 生成要删除或重复的单词的索引列表
    indices = random.sample(range(len(words)), num_to_modify)
    # 利用列表推导式生成新的单词列表，不包括要删除的单词，重复指定次数的单词添加到新列表中
    new_words = []
    for i, word in enumerate(words):
        if i not in indices:
            new_words.append(word)
        else:
            # 随机选择是删除还是重复
            if random.random() < 0.5:
                # 随机生成重复的次数，最多重复 3 次
                repeat_times = random.randint(1, 3)
                new_words.extend([word] * repeat_times)
    # 将新单词列表重新组合成字符串
    new_str = " ".join(new_words)
    return new_str

class T2TDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 max_source_length: int = 256,
                 max_target_length: int = 96,
                 start_idx: int = 106,
                 num_words: int = 1800,
                 eos_id: int = 105,
                 augment_text: bool = False):
        self.data = []
        with open(file_path) as fin:
            for line in fin:
                self.data.append(json.loads(line))
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.mapper = TextMapProcessor(start_idx, num_words, eos_id)
        self.augment_text = augment_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        text, summary = self.data[index]['text'], self.data[index]['summary']
        if self.augment_text:
            text = augment_text_fn(text)

        inputs = self.mapper.encode_t5([text],
                                       max_length=self.max_source_length,
                                       add_special_tokens=True, pad=True)
        labels = self.mapper.encode_t5([summary],
                                       max_length=self.max_target_length,
                                       add_special_tokens=True, pad=True)
        labels['input_ids'] = [
                [(l if l != 0 else -100) for l in label] for label in labels["input_ids"]
        ]
        return {
            'input_ids': torch.LongTensor(inputs['input_ids'][0]),
            'attention_mask': torch.LongTensor(inputs['attention_mask'][0]),
            'labels': torch.LongTensor(labels['input_ids'][0])
        }
