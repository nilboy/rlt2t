from typing import List
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict

from rlt2t.engines.t2t_engine import T2TEngineCT2
#from rlt2t.engines.reg_engine import RegEngine
from rlt2t.utils.evaluate import calculate_scores
import time

def sort_with_index(lst):
    indexed_lst = [(val, idx) for idx, val in enumerate(lst)]
    sorted_lst = sorted(indexed_lst)
    return [idx for val, idx in sorted_lst]

class Predictor(object):
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

    def calculate_score_for_flat_records_v1(self, records):
        output_records = []
        for item in records:
            output_records.append({
                'input': item['input'],
                'output': item['output'],
                'score_list': []
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
            item['score'] = np.mean(item['score_list'])
            del item['score_list']
        return output_records

    def calculate_score_for_flat_records_v2(self, records):
        output_records = []
        for item in records:
            output_records.append({
                'input': item['input'],
                'output': item['output'],
                'score_list': [],
            })
        for m_id, model_path in tqdm(enumerate(self.t2t_score_model_paths), 'calculate score...'):
            model_weight = self.score_model_weights[m_id]
            engine = T2TEngineCT2(model_path,
                                  compute_type="int8",
                                  num_words=self.num_words)
            scores = engine.get_scores(records, batch_size=self.batch_size,
                                       return_tokens_level=True)

            for i in range(len(scores)):
                output_records[i]['score_list'].append(scores[i])

        for record in output_records:
            # [model_num, tokens_num]
            log_probs = np.array(record['score_list'])
            # [tokens_num,]
            probs = np.mean(np.exp(log_probs), axis=0)
            score = np.sum(np.log(probs + 1e-8))/(len(probs) ** self.length_penalty)
            record['score'] = score
            del record['score_list']

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

    def predict_v2(self,
                   texts: List[str],
                   score_version='v2',
                   self_boost=False,
                   boost_size=4):
        """
        Return:
            output_records: List[Dict]
            {
                "input": "xx",
                "output": "xx",
                "score": 0.4
            }
        """
        texts_map = {}
        for text in texts:
            texts_map[text] = {'output_texts': []}
        for model_path in tqdm(self.t2t_model_paths, desc='beam_search_predict...'):
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
                    for beam_id, output_text in enumerate(output_texts[i]):
                        if output_text not in texts_map[text]['output_texts']:
                            texts_map[text]['output_texts'].append(output_text)

        # flat all records.
        flat_records = []
        for text, value in texts_map.items():
            output_texts = value['output_texts']
            for output_text in output_texts:
                flat_records.append({
                    'input': text, 'output': output_text
                })
        for k, v in texts_map.items():
            texts_map[k] = []
        print(f'total flat records: {len(flat_records)}')
        # calculate score for flat records.
        if score_version == 'v1':
            flat_records = self.calculate_score_for_flat_records_v1(flat_records)
        else:
            flat_records = self.calculate_score_for_flat_records_v2(flat_records)

        for flat_record in flat_records:
            texts_map[flat_record['input']].append({
                'output': flat_record['output'],
                'score':  flat_record['score']
            })
        if not self_boost:
            for text, output_list in texts_map.items():
                texts_map[text] = sorted(output_list, key=lambda x: -x['score'])[0]

            # 输出结果.
            output_records = []
            for text in texts:
                output_records.append({
                    'input': text,
                    'output': texts_map[text]['output'],
                    'score':  texts_map[text]['score']
                })
            return output_records
        else:
            for text, output_list in texts_map.items():
                texts_map[text] = sorted(output_list, key=lambda x: -x['score'])[0:boost_size]
            output_records = self.boost_records(texts_map, texts)
            return output_records

    def boost_records(self, texts_map, texts):
        flat_preds, flat_refs, flat_inputs = [], [], []
        for input_text, items in texts_map.items():
            for i in range(0, len(items)):
                for j in range(0, len(items)):
                    if i != j:
                        flat_inputs.append(input_text)
                        flat_preds.append(items[i]['output'])
                        flat_refs.append(items[j]['output'])
        # calculate scores
        t1 = time.time()
        score_list = calculate_scores(flat_refs, flat_preds, bleu4_rate=0.0)
        t2 = time.time()
        print(f'self-boost time: {t2 - t1} s')
        score_map = defaultdict(list)
        for i in range(len(flat_preds)):
            score_map[f"{flat_inputs[i]}_{flat_preds[i]}"].append(score_list[i])
        for k, v in score_map.items():
            score_map[k] = np.mean(v)
        for input_text, items in texts_map.items():
            for item in items:
                item['score'] = score_map[f"{input_text}_{item['output']}"]

        for text, output_list in texts_map.items():
            texts_map[text] = sorted(output_list, key=lambda x: -x['score'])[-1]
        # 输出结果.
        output_records = []
        for text in texts:
            output_records.append({
                'input': text,
                'output': texts_map[text]['output'],
                'score':  texts_map[text]['score']
            })
        return output_records

