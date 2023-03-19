import ctranslate2
import os
from rlt2t.textmap.textmap_processor import TextMapProcessor

class CLMEngineCT2(object):
    def __init__(self,
                 model_path,
                 compute_type='default',
                 device='cuda',
                 start_idx=106, eos_id=105, bos_id=104, num_words=7000):
        self.mapper = TextMapProcessor(start_idx=start_idx,
                                       eos_id=eos_id,
                                       bos_id=bos_id, num_words=num_words)
        self.ct2_model = ctranslate2.Generator(model_path,
                                               compute_type=compute_type,
                                               device=device)
        self.id2token, self.token2id = {}, {}
        with open(os.path.join(model_path, 'vocabulary.txt')) as fin:
            for idx, token in enumerate(fin):
                token = token.strip()
                self.id2token[idx] = token
                self.token2id[token] = idx

    def predict_records(self,
                        batch_size=64,
                        min_length=1,
                        max_length=128,
                        sampling_topk=10,
                        sampling_temperature=1.0,
                        **kwargs):
        inputs = [[self.id2token[self.mapper.bos_id]]] * batch_size
        results = self.ct2_model.generate_batch(inputs,
                                                min_length=min_length,
                                                max_length=max_length,
                                                sampling_topk=sampling_topk,
                                                sampling_temperature=sampling_temperature,
                                                **kwargs)
        preds = []
        for i in range(batch_size):
            tids = results[i].sequences_ids[0]
            filtered_tids = []
            for tid in tids:
                if tid == self.mapper.eos_id:
                    break
                if tid > 0:
                    filtered_tids.append(tid)
            output_text = self.mapper.decode(filtered_tids)
            preds.append(output_text)
        return preds
