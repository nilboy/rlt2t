import sys
import os
# 获取当前文件所在的文件夹路径
current_folder = os.path.dirname(os.path.abspath(__file__))
# 将当前文件所在的文件夹路径添加到Python包路径
sys.path.append(current_folder)

import pandas as pd
from rlt2t.predictor.predictor import Predictor

def get_t2t_model_paths():
    paths = []
    paths.append(os.path.join(current_folder, 'sub-models', 'uer-large-199-0.2'))
    return paths

def get_t2t_score_model_paths():
    paths = []
    paths.append(os.path.join(current_folder, 'sub-models', 'uer-large-199-0.2'))
    return paths

def invoke(input_data_path, output_data_path):
    split_token = '1799'
    t2t_model_paths = get_t2t_model_paths()
    t2t_score_model_paths = get_t2t_score_model_paths()
    predictor = Predictor(t2t_model_paths,
                          [],
                          t2t_score_model_paths,
                          beam_size=4,
                          num_hypotheses=1)
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
    pass
