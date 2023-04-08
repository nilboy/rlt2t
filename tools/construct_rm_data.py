import os
import fire
import json
from rlt2t.engines.t2t_engine import T2TEngineCT2
from rlt2t.utils.evaluate import CiderD
from tqdm.auto import tqdm

def calculate_scores(label_list, pred_list):
    cd_scorer = CiderD(df='corpus', sigma=15)
    gts = {
        idx: [label]
        for idx, label in enumerate(label_list)
    }

    res = [
        {'image_id': idx, "caption": [pred]}
        for idx, pred in enumerate(pred_list)
    ]
    cider_score, cider_scores = cd_scorer.compute_score(gts, res)
    return cider_scores

def construct_rm_data_kfold(data_dir='data',
                            model_dir="/root/autodl-tmp/output-models/t2t/ct2",
                            kfold=0,
                            max_score=10,
                            use_gt=False, use_beam=False):
    print(f'k_fold: {kfold}, use_gt: {use_gt}, use_beam: {use_beam}')
    records = []
    for line in open(os.path.join(data_dir, 'base/train.json')):
        record = json.loads(line)
        records.append({
            'text': (record['clinical'] + " " + record['description']).strip(),
            'gt_summary': record['diagnosis']
        })
    engine = T2TEngineCT2(os.path.join(model_dir, f'{kfold}'),
                          compute_type="int8",
                          num_words=1800)
    texts = [item['text'] for item in records]
    if use_gt:
        output_texts = [[record['gt_summary']] for record in records]
    elif use_beam:
        output_texts = engine.predict_records(texts, batch_size=64,
                                              beam_size=4,
                                              max_input_length=256,
                                              max_decoding_length=96,
                                              num_hypotheses=4,
                                              length_penalty=1.0)
    else:
        output_texts = engine.predict_records(texts, batch_size=64,
                                              beam_size=1,
                                              max_input_length=256,
                                              max_decoding_length=96,
                                              length_penalty=1.0,
                                              num_hypotheses=4,
                                              sampling_topk=3,
                                              sampling_temperature=0.6)
    output_records = []
    for j in range(len(output_texts[0])):
        output_texts_inner = [item[j] for item in output_texts]
        gt_output_texts = [record['gt_summary'] for record in records]
        scores = calculate_scores(gt_output_texts, output_texts_inner)/max_score

        for i in range(len(texts)):
            output_records.append({
                'text': texts[i],
                'summary': output_texts_inner[i],
                'label': float(scores[i])
            })
    return output_records


def construct_rm_data(data_dir="data",
                      model_dir="/root/autodl-tmp/t2t/ct2",
                      kfold_num=5,
                      max_score=10, max_iters=10):
    keys = set()
    os.makedirs(os.path.join(data_dir, 'rm'), exist_ok=True)
    r_id = 0
    with open(os.path.join(data_dir, 'rm/data.json'), 'w') as fout:
        for it in tqdm(range(max_iters)):
            use_gt, use_beam = False, False
            if it == 0:
                use_gt = True
            elif it == 1:
                use_beam = True

            for kfold_id in range(0, kfold_num):
                c_records = construct_rm_data_kfold(data_dir=data_dir,
                                                    model_dir=model_dir,
                                                    kfold=kfold_id,
                                                    max_score=max_score,
                                                    use_gt=use_gt, use_beam=use_beam)
                if use_beam or use_gt:
                    c_records = c_records * 3
                for record in c_records:
                    key = record['text'] + ' ' + record['summary']
                    if key not in keys:
                        r_id += 1
                        keys.add(key)
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(construct_rm_data)