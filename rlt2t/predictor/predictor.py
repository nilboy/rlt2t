from typing import List
from tqdm.auto import tqdm

from rlt2t.engines.t2t_engine import T2TEngineCT2
from rlt2t.engines.reg_engine import RegEngine


class Predictor(object):
    def __init__(self,
                 t2t_model_paths: List[str],
                 rm_model_paths: List[str],
                 num_words: int = 1800,
                 max_src_len: int = 256,
                 max_tgt_len: int = 96,
                 batch_size: int = 32,
                 beam_size: int = 4,
                 num_hypotheses: int = 1,
                 use_beam_search: bool = True,
                 sample_size: int = 4,
                 use_sample: bool = False,
                 use_fp16: bool = True):
        self.t2t_model_paths = t2t_model_paths
        self.rm_model_paths = rm_model_paths
        self.num_words = num_words
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_hypotheses = num_hypotheses
        self.use_beam_search = use_beam_search
        self.sample_size = sample_size
        self.use_sample = use_sample
        self.use_fp16 = use_fp16

    def calculate_scores(self, records):
        output_records = []
        for item in records:
            output_records.append({
                'input': item['input'],
                'output': item['output'],
                'score': 0.0
            })
        for rm_model_path in self.rm_model_paths:
            engine = RegEngine(rm_model_path, fp16=self.use_fp16)
            scores = engine.predict(output_records, batch_size=16)
            for i in range(len(scores)):
                output_records[i]['score'] += scores[i]
        for item in output_records:
            item['score'] /= len(self.rm_model_paths)
        return output_records

    def _predict_texts(self, texts, model_path, is_sample):
        engine = T2TEngineCT2(model_path,
                              compute_type="int8",
                              num_words=self.num_words)
        if is_sample:
            output_texts = engine.predict_records(texts, batch_size=self.batch_size,
                                                  beam_size=1,
                                                  max_input_length=self.max_src_len,
                                                  max_decoding_length=self.max_tgt_len,
                                                  num_hypotheses=self.sample_size,
                                                  length_penalty=1.0,
                                                  sampling_topk=4,
                                                  sampling_temperature=0.6)
        else:
            output_texts = engine.predict_records(texts, batch_size=self.batch_size,
                                                  beam_size=self.beam_size,
                                                  max_input_length=self.max_src_len,
                                                  max_decoding_length=self.max_tgt_len,
                                                  num_hypotheses=self.num_hypotheses,
                                                  length_penalty=1.0)
        output_records_list = []
        for j in range(len(output_texts[0])):
            output_records = [
                {'input': texts[i], 'output': output_texts[i][j]}
                for i in range(len(texts))
            ]
            output_records = self.calculate_scores(output_records)
            output_records_list.append(output_records)
        return output_records_list

    def predict(self,
                texts: List[str],
                return_best: bool = True):
        """
        Return:
            if return_best:
                output_records: List[Dict]
                {
                    "input": "xx",
                    "output": "xx",
                    "score": 0.4
                }
            else:
                output_records_list: List[List[Dict]]
        """
        output_records_list = []
        if self.use_beam_search:
            for model_path in tqdm(self.t2t_model_paths, desc='beam_search_predict...'):
                items = self._predict_texts(texts,
                                            model_path, is_sample=False)
                output_records_list.extend(items)
        if self.use_sample:
            for model_path in tqdm(self.t2t_model_paths, desc='sample predict...'):
                items = self._predict_texts(texts,
                                            model_path, is_sample=False)
                output_records_list.extend(items)
        if not return_best:
            return output_records_list
        else:
            best_records = []
            for i in range(len(output_records_list[0])):
                best_score, best_record = -1e8, None
                for j in range(len(output_records_list)):
                    if output_records_list[j][i]['score'] > best_score:
                        best_record = output_records_list[j][i]
                        best_score = output_records_list[j][i]['score']
                best_records.append(best_record)
            return best_records
