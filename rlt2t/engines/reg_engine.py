import torch

from rlt2t.models.bert_model import BertModelForRegression
from rlt2t.textmap.textmap_processor import TextMapProcessor

class RegEngine(object):
    def __init__(self,
                 model_path,
                 fp16=False,
                 max_token_len=256,
                 start_idx=106, eos_id=105, bos_id=104, num_words=1800):
        self.mapper = TextMapProcessor(start_idx=start_idx,
                                       eos_id=eos_id, bos_id=bos_id, num_words=num_words)
        self.model = BertModelForRegression.load_from_checkpoint(model_path).bert.cuda()
        self.model.eval()
        self.max_token_len = max_token_len
        self.fp16 = fp16
        if self.fp16:
            self.model = self.model.half()

    def predict(self, records, batch_size=8):
        results = []
        with torch.no_grad():
            for i in range(0, len(records), batch_size):
                cur_records = records[i:i+batch_size]
                inputs = {
                    'input_ids': [],
                    'token_type_ids': [],
                    'attention_mask': []
                }
                for record in cur_records:
                    encoded_item = self.mapper._encode_mlm_single(record['text'], record['summary'],
                                                                  max_length=self.max_token_len,
                                                                  pad=True)
                    inputs['input_ids'].append(encoded_item['input_ids'])
                    inputs['token_type_ids'].append(encoded_item['token_type_ids'])
                    inputs['attention_mask'].append(encoded_item['attention_mask'])
                for k, v in inputs.items():
                    inputs[k] = torch.LongTensor(v).cuda()
                outputs = self.model(**inputs).logits.cpu()[:,0].numpy().tolist()
                results.extend(outputs)
            return results



