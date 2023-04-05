import json
import random

import torch
from torch.utils.data import Dataset

from rlt2t.textmap.textmap_processor import TextMapProcessor
from rlt2t.utils.t5_mask_processor import T5MaskProcessor


class T2TPretrainDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 max_source_length: int = 256,
                 max_target_length: int = 96,
                 start_idx: int = 106,
                 num_words: int = 1800,
                 eos_id: int = 105,
                 noise_density: float = 0.15,
                 mean_noise_span_length: float = 3.0):
        self.data = []
        with open(file_path) as fin:
            for line in fin:
                self.data.append(json.loads(line))
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.mapper = TextMapProcessor(start_idx, num_words, eos_id)
        self.t5_mask_processor = T5MaskProcessor(1, eos_id,
                                                 noise_density=noise_density,
                                                 mean_noise_span_length=mean_noise_span_length,
                                                 process_count=8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        text = self.data[index]['text']
        inputs = [text]
        model_inputs = self.mapper.encode_t5(inputs,
                                             max_length=self.max_source_length,
                                             add_special_tokens=False)

        output_input_ids, output_labels = self.t5_mask_processor(model_inputs['input_ids'])

        input_ids, labels = output_input_ids[0], output_labels[0]
        attention_mask = [1] * len(input_ids)
        self.mapper.pad_to_max_length(input_ids, self.max_source_length)
        self.mapper.pad_to_max_length(attention_mask, self.max_source_length)
        self.mapper.pad_to_max_length(labels, self.max_target_length)

        labels = [(l if l != 0 else -100) for l in labels]
        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(labels)
        }
