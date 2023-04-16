import json
import os
import re
import sys
import threading
import traceback
import time
from flask import Flask, request, Response
from loguru import logger
from readerwriterlock import rwlock
from rlt2t.engines.t2t_engine import T2TEngineCT2
from rlt2t.utils.evaluate import calculate_scores
import random


def get_ensemble_data(base_model='/root/autodl-tmp/ct2',
                      data_file='data/t2t/data.json'):
    records = []
    with open(data_file) as fin:
        for line in fin:
            records.append(json.loads(line))
    model_paths = []
    for filename in os.listdir(base_model):
        if os.path.isdir(os.path.join(base_model, filename)):
            model_paths.append(os.path.join(base_model, filename))
    input_texts = [item['text'] for item in records]
    target_texts = [item['summary'] for item in records]
    model_outputs = []
    model_outputs_scores = []
    for model_path in model_paths:
        engine = T2TEngineCT2(model_path,
                              compute_type="int8",
                              num_words=1800)
        output_texts = engine.predict_records(input_texts,
                                              batch_size=64,
                                              beam_size=4,
                                              max_input_length=256,
                                              max_decoding_length=96,
                                              length_penalty=1.0)
        output_texts = [item[0] for item in output_texts]
        scores = calculate_scores(target_texts,
                         output_texts, bleu4_rate=0.0)
        model_outputs.append(output_texts)
        model_outputs_scores.append(scores)
    ensemble_data = {}
    for i in range(len(records)):
        text = input_texts[i]
        item = {
            'text':    text,
            'summary': target_texts[i],
            'candidate_outputs': [model_outputs[j][i] for j in range(len(model_outputs))],
            'candidate_outputs_scores': [model_outputs_scores[j][i] for j in range(len(model_outputs))]
        }
        ensemble_data[text] = item
    return ensemble_data


app = Flask(__name__)

app.ensemble_data = get_ensemble_data(base_model=sys.argv[1])
import json

json.dump(app.ensemble_data,
          open('ensemble_data.json', 'w'), ensure_ascii=False, indent=4)


@app.route('/get_rank_data', methods=['POST'])
def predict():
    text = request.json['text']
    example = app.ensemble_data[text]
    idx1, idx2 = random.sample(range(0, len(example['candidate_outputs'])),
                               2)
    if example['candidate_outputs_scores'][idx1] < example['candidate_outputs_scores'][idx2]:
        idx1, idx2 = idx2, idx1
    output_example = {
        'text': text,
        'summary': example['summary'],
        'rank_a': example['candidate_outputs'][idx1],
        'rank_b': example['candidate_outputs'][idx2],
        'rank_a_score': example['candidate_outputs_scores'][idx1],
        'rank_b_score': example['candidate_outputs_scores'][idx2],
        'use_rank': 1.0
    }
    if output_example['rank_a'] == output_example['rank_b'] or \
        output_example['rank_a_score'] == output_example['rank_b_score']:
        output_example['use_rank'] = 0.0
    return Response(json.dumps(output_example,
                               ensure_ascii=False, indent=4), status=200)


if __name__ == '__main__':
    logger.info("rank data 服务... ...")
    app.run(host="0.0.0.0", port=int(sys.argv[2]), debug=True, use_reloader=False)
