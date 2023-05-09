import pandas as pd
from rlt2t.predictor.predictor_score import PredictorScore
import json
import os

os.makedirs('data/rank', exist_ok=True)

input_file = "data/t2t/train_2_train/test.json"
df = pd.read_json(input_file, lines=True)
records = df.to_dict('records')

texts = [item['text'] for item in records]
summarys = [item['summary'] for item in records]

paths = [
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
 'uer-bart-large-1-rank']


t2t_model_paths = []
for item in paths:
    t2t_model_paths.append(os.path.join("/media/data1/jiangxinghua", 'sub-models', item))

paths = [
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

t2t_score_model_paths = []
for item in paths:
    t2t_score_model_paths.append(os.path.join("/media/data1/jiangxinghua", 'sub-models', item))

predictor = PredictorScore(t2t_model_paths,
                           [],
                           t2t_score_model_paths,
                           beam_size_list=[10],
                           num_hypotheses=4)
outputs, record_model_map = predictor.predict_score(texts, summarys, with_model_name=True)

with open('data/rank/train1.json', 'w') as fout:
    json.dump(outputs, fout, ensure_ascii=False, indent=2)

with open('data/rank/record_model_map.json', 'w') as fout:
     json.dump(record_model_map, fout, ensure_ascii=False, indent=2)
