from typing import List
from tqdm.auto import tqdm
import numpy as np

from rlt2t.engines.t2t_engine import T2TEngineCT2
#from rlt2t.engines.reg_engine import RegEngine
from rlt2t.utils.evaluate import calculate_scores
from collections import defaultdict

def sort_with_index(lst):
    indexed_lst = [(val, idx) for idx, val in enumerate(lst)]
    sorted_lst = sorted(indexed_lst)
    return [idx for val, idx in sorted_lst]

class PredictorScore(object):
    def __init__(self,
                 t2t_model_paths: List[str],
                 rm_model_paths: List[str],
                 t2t_score_model_paths: List[str]=None,
                 score_model_weights: List[float]=None,
                 num_words: int = 1800,
                 max_src_len: int = 256,
                 max_tgt_len: int = 96,
                 batch_size: int = 32,
                 beam_size: int = 4,
                 beam_size_list: List[int] = None,
                 num_hypotheses: int = 1,
                 length_penalty: float = 1.0,
                 sample_size: int = 1,
                 use_fp16: bool = True):
        self.t2t_model_paths = t2t_model_paths
        self.rm_model_paths = rm_model_paths
        self.t2t_score_model_paths = t2t_score_model_paths if t2t_score_model_paths else self.t2t_model_paths
        self.num_words = num_words
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.beam_size_list =beam_size_list
        self.num_hypotheses = num_hypotheses
        self.sample_size = sample_size
        self.use_fp16 = use_fp16
        self.length_penalty = length_penalty
        if score_model_weights is None:
            self.score_model_weights = [1.0] * len(self.t2t_score_model_paths)
        else:
            assert len(score_model_weights) == len(self.t2t_score_model_paths), 'weights lengths not matched'
            self.score_model_weights = score_model_weights

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
            item['score'] /= max(len(self.rm_model_paths), 1.0)
        return output_records

    def calculate_scores_with_gen_model(self, records):
        output_records = []
        for item in records:
            output_records.append({
                'input': item['input'],
                'output': item['output'],
                'score': []
            })
        for model_path in self.t2t_score_model_paths:
            engine = T2TEngineCT2(model_path,
                                  compute_type="int8",
                                  num_words=self.num_words)
            scores = engine.get_scores(records, batch_size=self.batch_size)
            for i in range(len(scores)):
                output_records[i]['score'].append(scores[i])
        for item in output_records:
            score_item = sorted(item['score'])
            #score_item = score_item[1:-1]
            #item['score'] /= max(len(self.t2t_model_paths), 1.0)
            if len(score_item) > 0:
                item['score'] = np.mean(score_item)
            else:
                item['score'] = 0.0

        return output_records

    def _predict_texts(self, texts, model_path, is_sample,
                       use_reward_model=True):
        engine = T2TEngineCT2(model_path,
                              compute_type="int8",
                              num_words=self.num_words)
        if is_sample:
            output_texts = engine.predict_records(texts, batch_size=self.batch_size,
                                                  beam_size=1,
                                                  max_input_length=self.max_src_len,
                                                  max_decoding_length=self.max_tgt_len,
                                                  num_hypotheses=self.sample_size,
                                                  length_penalty=self.length_penalty,
                                                  sampling_topk=4,
                                                  sampling_temperature=0.6)
        else:
            output_texts = engine.predict_records(texts, batch_size=self.batch_size,
                                                  beam_size=self.beam_size,
                                                  max_input_length=self.max_src_len,
                                                  max_decoding_length=self.max_tgt_len,
                                                  num_hypotheses=self.num_hypotheses,
                                                  length_penalty=self.length_penalty)
        output_records_list = []
        for j in range(len(output_texts[0])):
            output_records = [
                {'input': texts[i], 'output': output_texts[i][j]}
                for i in range(len(texts))
            ]
            if use_reward_model:
                output_records = self.calculate_scores(output_records)
            else:
                output_records = self.calculate_scores_with_gen_model(output_records)
            output_records_list.append(output_records)
        return output_records_list

    def predict(self,
                texts: List[str],
                return_best: bool = True,
                use_reward_model: bool = True,
                use_beam_search: bool = True,
                use_sample: bool = False):
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
        if use_beam_search:
            for model_path in tqdm(self.t2t_model_paths, desc='beam_search_predict...'):
                items = self._predict_texts(texts,
                                            model_path, is_sample=False,
                                            use_reward_model=use_reward_model)
                output_records_list.extend(items)
        if use_sample:
            for model_path in tqdm(self.t2t_model_paths, desc='sample predict...'):
                items = self._predict_texts(texts,
                                            model_path, is_sample=True, use_reward_model=use_reward_model)
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

    def calculate_score_for_flat_records(self, records):
        output_records = []
        for item in records:
            output_records.append({
                'input': item['input'],
                'output': item['output'],
                'score_list': [],
                'label': item['label'],
                'metric_score': item['metric_score']
            })
        for m_id, model_path in tqdm(enumerate(self.t2t_score_model_paths), 'calculate score...'):
            model_weight = self.score_model_weights[m_id]
            engine = T2TEngineCT2(model_path,
                                  compute_type="int8",
                                  num_words=self.num_words)
            scores = engine.get_scores(records, batch_size=self.batch_size,
                                       return_tokens_level=True)
            for i in range(len(scores)):
                output_records[i]['score_list'] \
                    .append(np.sum(scores[i])/(len(scores[i]) ** self.length_penalty) * model_weight)
        for item in output_records:
            item['score'] = item['score_list']
            del item['score_list']
        return output_records

    def get_best_output(self, output_list):
        """
        [
            {
                "output": "xxx",
                "score_list": List[float]
            }
        ]
        """
        for item in output_list:
            item['rank'] = []
        for model_id in range(len(output_list[0]['score_list'])):
            model_scores = [item['score_list'][model_id] for item in output_list]
            rank_list = sort_with_index(model_scores)
            for i in range(len(output_list)):
                output_list[i]['rank'].append(rank_list[i])
        merged_output = []
        for item in output_list:
            merged_output.append({
                'output': item['output'],
                'rank': np.mean(item['rank']),
                'score': np.mean(item['score_list'])
            })
        merged_output.sort(key=lambda x: (-x['rank'], -x['score']))
        return merged_output[0]

    def predict_score(self,
                   texts: List[str],
                   labels: List[str], with_model_name=False):
        """
        Return:
            output_records: List[Dict]
            {
                "input": "xx",
                "output": "xx",
                "score": 0.4
            }
        """

        record_model_map = defaultdict(list)
        texts_map = {}
        for idx, text in enumerate(texts):
            texts_map[text] = {
                'output_texts': [],
                'label': labels[idx]
            }
        for model_path in tqdm(self.t2t_model_paths, desc='beam_search_predict...'):
            model_name = model_path.split('/')[-1]
            engine = T2TEngineCT2(model_path,
                                  compute_type="int8",
                                  num_words=self.num_words)
            if self.beam_size_list is not None:
                beam_size_list = self.beam_size_list
            else:
                beam_size_list = [self.beam_size]
            for beam_size in beam_size_list:
                output_texts = engine.predict_records(texts, batch_size=self.batch_size,
                                                      beam_size=beam_size,
                                                      max_input_length=self.max_src_len,
                                                      max_decoding_length=self.max_tgt_len,
                                                      num_hypotheses=min(self.num_hypotheses, beam_size),
                                                      length_penalty=self.length_penalty)
                for i, text in enumerate(texts):
                    for output_text in output_texts[i]:
                        record_model_map[f'{text}_{output_text}'].append((beam_size, model_name))
                        if output_text not in texts_map[text]['output_texts']:
                            texts_map[text]['output_texts'].append(output_text)

        # flat all records.
        flat_records = []
        for text, value in texts_map.items():
            output_texts = value['output_texts']
            label = value['label']
            for output_text in output_texts:
                flat_records.append({
                    'input': text,
                    'output': output_text,
                    'label': label,
                    'metric_score': -1.0
                })
        # calculate metric_score.
        metric_scores = calculate_scores([item['label'] for item in flat_records],
                                         [item['output'] for item in flat_records],
                                         bleu4_rate=1.0 / 3)
        for i in range(len(flat_records)):
            flat_records[i]['metric_score'] = metric_scores[i]
        # calculate score for flat records.
        flat_records = self.calculate_score_for_flat_records(flat_records)
        for k, v in texts_map.items():
            texts_map[k] = []
        for flat_record in flat_records:
            texts_map[flat_record['input']].append({
                'output': flat_record['output'],
                'metric_score': flat_record['metric_score'],
                'score': flat_record['score']
            })
        if not with_model_name:
            return texts_map
        else:
            return texts_map, record_model_map

