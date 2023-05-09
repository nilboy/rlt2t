import copy
import json
import numpy as np
from tqdm.auto import tqdm

from joblib import Parallel, delayed

records = json.load(open('data/rank/train1.json'))

model_num = None
for k, v in records.items():
    model_num = len(v[0]['score'])
    break

def get_score(model_weights):
    sum_score = 0
    model_weights = np.array(model_weights)
    for k, item_list in records.items():
        max_score = -100000
        metric_score = 0.0
        for item in item_list:
            cur_score = np.sum(np.array(item['score']) * model_weights)
            if cur_score > max_score:
                max_score = cur_score
                metric_score = item['metric_score']
        sum_score += metric_score
    return sum_score / len(records), [int(item) for item in model_weights]

model_names = [
'idea-bart-xl-0.2-rank',
 'idea-bart-base',
 'uer-base-139-0.1-142-rank',
 'fnlp-base-249-242-503650-rank',
 'idea-bart-xl-2-rank',
 'uer-bart-large-1-rank',
 'uer-large-199-0.1-rank',
 'uer-large-199-0.2',
 'uer-large-199-0.2-rank',
 'idea-bart-base-rank',
 'idea-bart-xl-1-rank',
 'idea-bart-xl-1',
 'uer-large-199-0.1',
 'idea-bart-xl-0.3',
 'idea-bart-xl-0.2'
]


import itertools

max_score = 0.0

with open('records.jsonl', 'w') as fout:
    for start_model_id in range(0, 15, 15):
        results = Parallel(n_jobs=30)(delayed(get_score)([0] * start_model_id + list(model_flags) + [0] * (len(model_names) - start_model_id - 15)) for model_flags in tqdm(itertools.product([0, 1], repeat=15)))

        for item in results:
            try:
                fout.write(json.dumps({'score': item[0], 'model_flags': item[1]}, ensure_ascii=False) + '\n')
            except:
                import ipdb
                ipdb.set_trace()




