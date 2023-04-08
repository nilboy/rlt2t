import sys
import os
import fire
import json
from rlt2t.engines.t2t_engine import T2TEngineCT2, T2TEngineHF
from rlt2t.predictor.predictor import Predictor

def generate_weak_data(data_dir='data/weak_t2t',
                       model_dir="/root/autodl-tmp/ct2",
                       rm_model_path="/root/autodl-tmp/score-model"):
    t2t_model_paths = []
    for name in os.listdir(model_dir):
        full_name = os.path.join(model_dir, name)
        if os.path.isdir(full_name):
            t2t_model_paths.append(full_name)
    rm_model_paths = [rm_model_path]
    predictor = Predictor(t2t_model_paths, rm_model_paths, use_beam_search=True, beam_size=4, num_hypotheses=1)
    texts = []
    for line in open(os.path.join(data_dir, 'gen_clm.json')):
        texts.append(json.loads(line)['text'])
    output_records = predictor.predict(texts, return_best=True)
    with open(os.path.join(data_dir, 'data.json'), 'w') as fout:
        for record in output_records:
            fout.write(json.dumps(
                {"text": record['input'], 'summary': record['output'], 'score': record['score']}, ensure_ascii=False
            ) + '\n')

if __name__ == '__main__':
    fire.Fire(generate_weak_data)