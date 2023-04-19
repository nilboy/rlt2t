import fire
import json
from itertools import combinations
import random

def get_combinations(lst):
    return list(combinations(lst, 2))


def construct_mlm_data_file(input_file,
                            output_file,
                            limit_size=None):
    records = []
    for line in open(input_file):
        records.append(json.loads(line))
    output_records = []
    for record in records:
        text = record['text']
        for summary_item in record['summary_list']:
            output_records.append({
                'text': text,
                'summary': summary_item['summary']
            })
    random.shuffle(output_records)
    if limit_size:
        output_records = output_records[0:limit_size]
    with open(output_file, 'w') as fout:
        for record in output_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def construct_mlm_data():
    construct_mlm_data_file('data/score/train.json', 'data/mlm/train.json')
    construct_mlm_data_file('data/score/test.json', 'data/mlm/test.json', 2000)


if __name__ == '__main__':
    fire.Fire(construct_mlm_data)