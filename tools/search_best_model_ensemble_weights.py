import copy
import json
import numpy as np
from tqdm.auto import tqdm

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
    return sum_score / len(records)

model_names = [
        'idea-bart-xl-2-rank',
        'fnlp-base-249-242-503650-rank',
        'idea-bart-xl-0.2-rank',
        'uer-large-199-0.1-rank',
        'uer-base-139-0.1-142-rank',
        'idea-bart-xl-1-rank',
        'uer-large-199-0.1',
        'idea-bart-xl-0.3',
        'uer-large-199-0.2',
        'idea-bart-base-rank',
        'idea-bart-base',
        'uer-bart-large-1-rank'
    ]

model_weights = [0.0] * len(model_names)

choise_models = [
    'idea-bart-xl-2-rank',
    "idea-bart-xl-0.2-rank",
    "uer-large-199-0.1-rank",
    'uer-base-139-0.1-142-rank',
    "idea-bart-base-rank",
    "fnlp-base-249-242-503650-rank",
]

model_weights[model_names.index(choise_models[0])] = 1.0

# # load json
# old_weight = json.load(open('weights.json'))
# for k, v in old_weight.items():
#     model_weights[model_names.index(k)] = v

def search_single_model(model_id):
    min_w, max_w = min(model_weights), max(model_weights)
    w_range = max(max_w - min_w, 1.0)
    min_w, max_w = max(min_w - w_range / 2, 0), max_w + w_range / 2
    max_score_w, max_score = 0, -np.inf
    for w in tqdm([model_weights[model_id]] + np.arange(min_w, max_w, (max_w - min_w) / 10).tolist()):
        cur_model_weights = copy.deepcopy(model_weights)
        cur_model_weights[model_id] = w
        cur_score = get_score(cur_model_weights)
        if cur_score > max_score:
            max_score = cur_score
            max_score_w = w
    model_weights[model_id] = max_score_w
    return max_score


for cur_choise_model_num in range(2, len(choise_models) + 1):
    cur_choise_models = choise_models[0:cur_choise_model_num]
    # choise cur_last_model
    last_model_id = model_names.index(choise_models[cur_choise_model_num - 1])
    max_score_w = search_single_model(last_model_id)
    print(f"choise_model_num: {cur_choise_model_num}, max_score: {max_score_w}")
    for i in range(3):
        for model_id in tqdm(range(model_num - 1, -1, -1)):
            if model_names[model_id] not in cur_choise_models:
                continue
            max_score_w = search_single_model(model_id)
            print(f"choise_model_num: {cur_choise_model_num}, max_score: {max_score_w}")
    print(model_weights)

model_kv = {}
for i in range(len(model_names)):
    model_kv[model_names[i]] = model_weights[i]

with open('weights.json', 'w') as fout:
    json.dump(model_kv, fout, ensure_ascii=False, indent=2)



