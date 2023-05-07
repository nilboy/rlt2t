import copy
import json
import numpy as np
from tqdm.auto import tqdm

records = json.load(open('data/rank/train.json'))

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
    return sum_score / len(records)

model_names = [
     'uer-large-199-0.2-rank',
     'uer-large-199-0.2',
     'uer-large-199-0.1-rank',
     'uer-large-199-0.1',
     'uer-pegasus-large-rank',
     'idea-pegasus-large-rank',
     'idea-pegasus-base-rank',
     'uer-pegasus-large',
     'idea-pegasus-large',
     'idea-bart-base-rank',
     'idea-pegasus-base',
     'uer-pegasus-base-rank',
     'idea-bart-base',
     'uer-pegasus-base',
     'uer-base-139-0.1-142-rank',
     'fnlp-base-249-242-503650-rank',
     'idea-bart-xl-0.2-rank',
     'idea-bart-xl-0.3',
     'idea-bart-xl-0.2',
     'uer-base-139-0.1-188',
     'fnlp-base-249-242-503657',
     'fnlp-base-249-242-503650',
     'uer-base-139-0.1-142',
]

model_weights = [0.0] * len(model_names)

choise_models = [
    'idea-bart-xl-0.2',
    'uer-large-199-0.2',
    'uer-large-199-0.1-rank',
    'idea-bart-base-rank',
    'uer-large-199-0.1',
    'uer-large-199-0.2-rank',
    'uer-base-139-0.1-142-rank',
    'idea-bart-xl-0.2-rank',
    'idea-bart-xl-0.3',
    'uer-base-139-0.1-142',
]

model_weights[model_names.index(choise_models[0])] = 1.0

for c_model_num in range(2, len(choise_models) + 1):
    cur_choise_models = choise_models[0:c_model_num]
    for i in range(3):
        for model_id in tqdm(range(model_num-1, -1, -1)):
            if model_names[model_id] not in cur_choise_models:
                continue
            min_w, max_w = min(model_weights), max(model_weights)
            w_range = max(max_w - min_w, 1.0)
            #min_w, max_w = max(min_w - w_range / 2 * 5, 0.0), max_w + w_range / 2 * 5
            min_w, max_w = min_w - w_range / 2, max_w + w_range / 2
            print(min_w, max_w)
            max_score_w, max_score = 0, -np.inf
            for w in tqdm(np.arange(min_w, max_w, (max_w - min_w)/100)):
                cur_model_weights = copy.deepcopy(model_weights)
                cur_model_weights[model_id] = w
                cur_score = get_score(cur_model_weights)
                if cur_score > max_score:
                    max_score = cur_score
                    max_score_w = w
            print(model_id, max_score_w, max_score)
            model_weights[model_id] = max_score_w
        print(model_weights)

model_kv = {}
for i in range(len(model_names)):
    model_kv[model_names[i]] = model_weights[i]

with open('weights.json', 'w') as fout:
    json.dump(model_kv, fout, ensure_ascii=False, indent=2)



