import ctranslate2
import os
import transformers
from transformers import AutoModelForSeq2SeqLM
from rlt2t.textmap.textmap_processor import TextMapProcessor
from typing import List
from tqdm.auto import tqdm
import torch

class T2TEngineCT2(object):
    def __init__(self,
                 model_path,
                 compute_type='default',
                 device='cuda',
                 start_idx=106, eos_id=105):
        self.mapper = TextMapProcessor(start_idx=start_idx,
                                       eos_id=eos_id)
        self.ct2_model = ctranslate2.Translator(model_path,
                                                compute_type=compute_type,
                                                device=device)
        self.id2token, self.token2id = {}, {}
        with open(os.path.join(model_path, 'shared_vocabulary.txt')) as fin:
            for idx, token in enumerate(fin):
                token = token.strip()
                self.id2token[idx] = token
                self.token2id[token] = idx

    def predict_records(self, records: List[str], batch_size=64,
                        beam_size=2,
                        max_input_length=1024,
                        max_decoding_length=256,
                        sampling_topk=1,
                        sampling_temperature=1.0):
        preds, labels = [], []
        for i in tqdm(range(0, len(records), batch_size),
                      desc="predict..."):
            cur_records = records[i:i + batch_size]
            inputs = [
                [self.id2token[item] for item in self.mapper.encode_t5([record], max_input_length)['input_ids'][0]]
                for record in cur_records
            ]
            results = self.ct2_model.translate_batch(inputs,
                                                     beam_size=beam_size,
                                                     max_input_length=max_input_length,
                                                     max_decoding_length=max_decoding_length,
                                                     sampling_topk=sampling_topk,
                                                     sampling_temperature=sampling_temperature)
            for j in range(0, len(cur_records)):
                target = results[j].hypotheses[0]
                filtered_tids = []
                tids = [self.token2id[item] for item in target]
                for tid in tids:
                    if tid == self.mapper.eos_id:
                        break
                    if tid > 0:
                        filtered_tids.append(tid)
                output_text = self.mapper.decode(filtered_tids)
                preds.append(output_text)
        return preds

class T2TEngineHF(object):
    def __init__(self,
                 model_path,
                 device='cuda:0',
                 start_idx=106, eos_id=105):
        self.mapper = TextMapProcessor(start_idx=start_idx,
                                       eos_id=eos_id)
        self.device = device
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    def predict_records(self, records: List[str], batch_size=64,
                        beam_size=2,
                        max_input_length=1024,
                        max_decoding_length=256,
                        sampling_topk=1,
                        sampling_temperature=1.0,
                        do_sample=False):
        preds = []
        for i in tqdm(range(0, len(records), batch_size), desc='predict...'):
            cur_records = records[i:i + batch_size]
            x = self.mapper.encode_t5([item for item in cur_records], max_input_length)
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            max_len = max([len(item) for item in input_ids])
            for j in range(len(cur_records)):
                if len(input_ids[j]) < max_len:
                    pad_len = max_len - len(input_ids[j])
                else:
                    pad_len = 0
                input_ids[j].extend([0] * pad_len)
                attention_mask[j].extend([0] * pad_len)
            with torch.no_grad():
                outputs = self.model.generate(input_ids=torch.tensor(input_ids).to(self.device),
                                              attention_mask=torch.tensor(attention_mask).to(self.device),
                                              num_beams=beam_size,
                                              max_length=max_decoding_length,
                                              top_k=sampling_topk,
                                              temperature=sampling_temperature,
                                              do_sample=do_sample)
            outputs = outputs.detach().cpu().tolist()

            for j in range(0, len(cur_records)):
                filtered_tids = []
                tids = outputs[j]
                for tid in tids:
                    if tid == self.mapper.eos_id:
                        break
                    if tid > 0:
                        filtered_tids.append(tid)
                output_text = self.mapper.decode(filtered_tids)
                preds.append(output_text)
        return preds