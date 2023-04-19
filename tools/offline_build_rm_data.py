import time
import json
from rlt2t.predictor.predictor import Predictor
import random
from rlt2t.utils.evaluate import calculate_scores
import sys


if __name__ == '__main__':
    mode = sys.argv[1]
    print(mode)
    if mode == 'train':
        fout = open("data/rm2/data.json", "w")
    else:
        fout = open("data/rm2/test_data.json", "w")

    t2t_model_paths = [
        "/root/autodl-tmp/ct2/0",
        "/root/autodl-tmp/ct2/1",
        "/root/autodl-tmp/ct2/2",
        "/root/autodl-tmp/ct2/3",
        "/root/autodl-tmp/ct2/4",
        "/root/autodl-tmp/ct2/base_17",
        "/root/autodl-tmp/ct2/base_19",
        "/root/autodl-tmp/ct2/01_13",
        "/root/autodl-tmp/ct2/01_19",
        "/root/autodl-tmp/ct2/05_19",
        "/root/autodl-tmp/ct2/05_29",
    ]
    rm_model_paths = []
    if mode == 'train':
        data_file = "/root/project/rlt2t/data/t2t/train.json"
        repeat_num = 200
    else:
        data_file = "/root/project/rlt2t/data/t2t/test.json"
        repeat_num = 2

    input_records = []
    for line in open(data_file):
        input_records.append(json.loads(line))

    for i in range(repeat_num):
        # 生成记录
        if mode == 'train':
            predictor = Predictor(random.sample(t2t_model_paths, 2),
                                  rm_model_paths, num_hypotheses=4, sample_size=2)
            output_records_list = predictor.predict([item['text'] for item in input_records],
                                                    return_best=False, use_beam_search=True, use_sample=True)
        else:
            predictor = Predictor(random.sample(t2t_model_paths, 2),
                                  rm_model_paths, num_hypotheses=1, sample_size=2)
            output_records_list = predictor.predict([item['text'] for item in input_records],
                                                    return_best=False, use_beam_search=True, use_sample=False)

        filtered_records = []
        for i in range(len(input_records)):
            output_texts = list(set([model_outputs[i]['output'] for model_outputs in output_records_list]))
            if len(output_texts) >= 2:
                output_text_a, output_text_b = random.sample(output_texts, 2)
                filtered_records.append({
                    'text': input_records[i]['text'],
                    'summary': input_records[i]['summary'],
                    'summary_a': output_text_a,
                    'summary_b': output_text_b
                })
        summary_list = [item['summary'] for item in filtered_records]
        summary_a_list = [item['summary_a'] for item in filtered_records]
        summary_b_list = [item['summary_b'] for item in filtered_records]
        scores_a = calculate_scores(summary_list, summary_a_list, bleu4_rate=0.0)
        scores_b = calculate_scores(summary_list, summary_b_list, bleu4_rate=0.0)
        for i in range(len(filtered_records)):
            filtered_records[i]['score_a'] = scores_a[i]
            filtered_records[i]['score_b'] = scores_b[i]
        # change a, b sort
        for i in range(len(filtered_records)):
            if filtered_records[i]['score_a'] < filtered_records[i]['score_b']:
                filtered_records[i]['summary_a'], filtered_records[i]['summary_b'] = \
                    filtered_records[i]['summary_b'], filtered_records[i]['summary_a']
                filtered_records[i]['score_a'], filtered_records[i]['score_b'] = \
                    filtered_records[i]['score_b'], filtered_records[i]['score_a']
        for item in filtered_records:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    fout.close()
