import sys
import os
# 获取当前文件所在的文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在的文件夹路径添加到Python包路径
sys.path.append(current_folder)

import pandas as pd
from rlt2t.predictor.predictor import Predictor

def get_t2t_model_paths():
    paths = [
        'uer-large-199-0.2-rank',
        'uer-large-199-0.2',
        'idea-pegasus-large',
        'idea-bart-base-rank',
        'uer-pegasus-base',
        'fnlp-base-249-242-503650-rank',
        'idea-bart-xl-0.2-rank',
        'uer-base-139-0.1-188',
        'fnlp-base-249-242-503650',
        'uer-base-139-0.1-142'
    ]

    # paths = [
    #      'uer-pegasus-large-rank',
    #      'idea-pegasus-large',
    #      'idea-bart-base-rank',
    #      'uer-pegasus-base-rank',
    #      'idea-bart-base',
    #      'uer-pegasus-base',
    #      'idea-bart-xl-0.2',
    #      'uer-base-139-0.1-188',
    #      'fnlp-base-249-242-503657',
    #      'uer-base-139-0.1-142'
    # ]

    output_paths = []
    for item in paths:
        output_paths.append(os.path.join(current_folder, 'sub-models', item))
    return output_paths

def get_t2t_score_model_paths():
    paths = [
        "uer-large-199-0.2",
        "uer-large-199-0.1-rank",
        "idea-bart-base-rank",
        "uer-base-139-0.1-142-rank",
        "fnlp-base-249-242-503650-rank",
        "idea-bart-xl-0.2",
        "fnlp-base-249-242-503650",
    ]
    weights = [
        3.1807214547847193,
        3.0532976036639994,
        2.0744802872639996,
        2.4505946106337073,
        0.14313246546531236,
        1.0915989281099199,
        0.27479678432975996,
    ]
    output_paths = []
    for item in paths:
        output_paths.append(os.path.join(current_folder, 'sub-models', item))
    return output_paths, weights

def invoke(input_data_path, output_data_path):
    split_token = '1799'
    t2t_model_paths = get_t2t_model_paths()
    t2t_score_model_paths, weights = get_t2t_score_model_paths()
    predictor = Predictor(t2t_model_paths,
                          [],
                          t2t_score_model_paths,
                          score_model_weights=weights,
                          beam_size_list=[1, 2, 4],
                          batch_size=128,
                          num_hypotheses=4)
    df = pd.read_csv(input_data_path,
                     header=None, index_col=False,
                     names=["report_id", "description", "clinical"])
    records = df.to_dict('records')
    for idx, record in enumerate(records):
        if not record.get('clinical', '') or str(record['clinical']) == 'nan':
            record['clinical'] = ''

    texts = []
    for record in records:
        text = " ".join([record['clinical'].strip(), split_token, record['description'].strip()])
        texts.append(text)
    output_texts = [item['output'] for item in predictor.predict_v2(texts)]

    output_records = []
    for idx in range(len(records)):
        output_records.append({
            'report_id': records[idx]['report_id'],
            'prediction': output_texts[idx]
        })
    output_df = pd.DataFrame(output_records)
    output_df.to_csv(output_data_path, header=False, index=False)


if __name__ == '__main__':
    import time
    t1 = time.time()
    invoke('test.csv', 'output.csv')
    t2 = time.time()
    print(t2-t1)
