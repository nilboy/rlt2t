import time
import json
from rlt2t.predictor.predictor import Predictor
import random
from rlt2t.utils.evaluate import calculate_scores
import os
import sys
import fire

def remove_duplicate_text(lst):
    seen = set()
    filtered_lst = []
    for d in lst:
        if d['summary'] not in seen:
            seen.add(d['summary'])
            filtered_lst.append(d)
    return filtered_lst

def construct_score_data_file(input_file, output_file,
                              bleu4_score=0.0):
    input_records = []
    for line in open(input_file):
        input_records.append(json.loads(line))
    t2t_model_paths = [
        "/root/autodl-tmp/ct2/0",
        "/root/autodl-tmp/ct2/1",
        "/root/autodl-tmp/ct2/2",
        "/root/autodl-tmp/ct2/3",
        "/root/autodl-tmp/ct2/4"
    ]
    rm_model_paths = []
    predictor = Predictor(t2t_model_paths, rm_model_paths, num_hypotheses=1)
    output_records_list = predictor.predict([item['text'] for item in input_records],
                                            return_best=False, use_beam_search=True, use_sample=False)
    summary_list = [item['summary'] for item in input_records]
    gt_scores = calculate_scores(summary_list, summary_list)
    output_records = [
                       {
                           "text": item['text'],
                           "summary_list": [
                               {
                                   "summary": item['summary'],
                                   "score": gt_scores[i]
                               }
                           ]
                       }
                      for i, item in enumerate(input_records)
    ]
    for model_id in range(len(output_records_list)):
        output_texts = [item['output'] for item in output_records_list[model_id]]
        scores = calculate_scores(summary_list, output_texts, bleu4_rate=bleu4_score)
        for j in range(len(input_records)):
            output_records[j]['summary_list'].append({
                'summary': output_texts[j],
                'score': scores[j]
            })
    # filter duplicate records
    filtered_records = []
    for record in output_records:
        dedup_items = remove_duplicate_text(record['summary_list'])
        if len(dedup_items) < 2:
            continue
        filtered_records.append({
            "text": record['text'],
            "summary_list": dedup_items
        })
    with open(output_file, 'w') as fout:
        for record in filtered_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def construct_score_data():
    os.makedirs('data/score', exist_ok=True)
    construct_score_data_file('data/t2t/test.json',
                              'data/score/test.json')
    construct_score_data_file('data/t2t/train.json',
                              'data/score/train.json')

if __name__ == '__main__':
    fire.Fire(construct_score_data)
