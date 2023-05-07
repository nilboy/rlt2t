import json

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

import numpy as np

records = json.load(open('data/rank/train1.json'))

record_model_map = json.load(open('data/rank/record_model_map.json'))

model_weights = json.load(open('weights_1.json'))

model_weights = np.array([model_weights[model_name] for model_name in model_names])

def get_score(score, model_weights):
     score = np.array(score)
     return np.sum(score * model_weights)

parsed_records = {}

for text, outputs in records.items():
     parsed_outputs = []
     for item in outputs:
          output_item = {
               'output': item['output'],
               'metric_score': item['metric_score'],
               'score': get_score(item['score'], model_weights),
               'models': set([model_name for b_s, model_name in record_model_map[text + '_' + item['output']]])
          }
          parsed_outputs.append(output_item)
     parsed_outputs.sort(key=lambda x: -x['score'])
     parsed_records[text] = parsed_outputs

import itertools

from tqdm.auto import tqdm

def get_ensemble_model_score(model_flags):
     used_models = set([model_names[i] for i in range(len(model_flags)) if model_flags[i] == 1])
     scores = []
     for k, outputs in parsed_records.items():
          for output_item in outputs:
               if used_models.intersection(output_item['models']):
                    scores.append(output_item['metric_score'])
                    break
     return np.mean(scores)

model_flags_scores = []

max_score = 0


with open('model_flags_scores.jsonl', 'w') as fout:
     for model_flags in tqdm(itertools.product([0, 1], repeat=len(model_names))):
          model_score = get_ensemble_model_score(model_flags)
          if model_score > max_score:
               max_score = model_score
               print(model_score)
               print(model_flags)
          fout.write(json.dumps({'score': model_score, 'model_flags': model_flags}, ensure_ascii=False) + '\n')

