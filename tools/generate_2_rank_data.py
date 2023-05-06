import pandas as pd
from rlt2t.predictor.predictor_score import PredictorScore
import json

input_file = "data/t2t/train_2_train/test.json"
df = pd.read_json(input_file, lines=True)
records = df.to_dict('records')

texts = [item['text'] for item in records]
summarys = [item['summary'] for item in records]

t2t_model_paths = [
    "/root/autodl-tmp/sub-models/uer-base-139-0.1-142-rank",
    "/root/autodl-tmp/sub-models/fnlp-base-249-242-503650-rank",
    "/root/autodl-tmp/sub-models/uer-pegasus-base-rank",
    "/root/autodl-tmp/sub-models/idea-bart-base-rank",
    # "/root/autodl-tmp/sub-models/uer-large-199-0.2-rank",
    # "/root/autodl-tmp/sub-models/idea-bart-xl-0.2-rank",
]

t2t_score_model_paths = [
    "/root/autodl-tmp/sub-models/uer-base-139-0.1-142-rank",
    "/root/autodl-tmp/sub-models/fnlp-base-249-242-503650-rank",
    "/root/autodl-tmp/sub-models/uer-pegasus-base-rank",
    "/root/autodl-tmp/sub-models/idea-bart-base-rank",
    # "/root/autodl-tmp/sub-models/uer-large-199-0.2-rank",
    # "/root/autodl-tmp/sub-models/idea-bart-xl-0.2-rank",
]

predictor = PredictorScore(t2t_model_paths,
                           [],
                           t2t_score_model_paths,
                           beam_size_list=[1, 2, 4],
                           num_hypotheses=4)
outputs = predictor.predict_score(texts, summarys)

with open('data/rank/train.json', 'w') as fout:
    json.dump(outputs, fout, ensure_ascii=False, indent=2)
