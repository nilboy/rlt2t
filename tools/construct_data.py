import random

import fire
import os
import pandas as pd
import random
import json
from loguru import logger


def split_train_test_data(input_data_dir,
                          output_data_dir, stage, num_kfold,
                          test_data_size):
    os.makedirs(os.path.join(output_data_dir, 'base'), exist_ok=True)
    if stage == 0:
        df = pd.read_csv(os.path.join(input_data_dir, 'train.csv'), header=None, index_col=False,
                         names=["report_id", "description", "diagnosis"])
    else:
        df = pd.read_csv(os.path.join(input_data_dir, 'train.csv'), header=None, index_col=False,
                         names=["report_id", "description", "diagnosis", "clinical"])
    records = df.to_dict('records')
    random.shuffle(records)
    test_records, train_records = [], []
    for idx, record in enumerate(records):
        if 'clinical' not in record:
            record['clinical'] = ''
        if idx < test_data_size:
            test_records.append(record)
        else:
            record['kfold_id'] = idx % num_kfold
            train_records.append(record)
    with open(os.path.join(output_data_dir, 'base/train.json'), 'w') as fout:
        for record in train_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    with open(os.path.join(output_data_dir, 'base/test.json'), 'w') as fout:
        for record in test_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def construct_clm_data(output_data_dir):
    os.makedirs(os.path.join(output_data_dir, 'clm'), exist_ok=True)
    train_data = pd.read_json(os.path.join(output_data_dir,
                                           'base/train.json'), lines=True)
    test_data = pd.read_json(os.path.join(output_data_dir,
                                          'base/test.json'), lines=True)
    with open(os.path.join(os.path.join(output_data_dir, 'clm/train.json')), 'w') as fout:
        for _, record in train_data.iterrows():
            output_record = {
                'text': (record['clinical'] + " " + record['description']).strip()
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    with open(os.path.join(os.path.join(output_data_dir, 'clm/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': (record['clinical'] + " " + record['description']).strip()
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')


def construct_mlm_data(output_data_dir):
    os.makedirs(os.path.join(output_data_dir, 'mlm'), exist_ok=True)
    train_data = pd.read_json(os.path.join(output_data_dir,
                                           'base/train.json'), lines=True)
    test_data = pd.read_json(os.path.join(output_data_dir,
                                          'base/test.json'), lines=True)
    with open(os.path.join(os.path.join(output_data_dir, 'mlm/train.json')), 'w') as fout:
        for _, record in train_data.iterrows():
            output_record = {
                'text': (record['clinical'] + " " + record['description']).strip(),
                'summary': record['diagnosis']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    with open(os.path.join(os.path.join(output_data_dir, 'mlm/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': (record['clinical'] + " " + record['description']).strip(),
                'summary': record['diagnosis']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')


def construct_pt2t_data(output_data_dir):
    os.makedirs(os.path.join(output_data_dir, 'pt2t'), exist_ok=True)
    train_data = pd.read_json(os.path.join(output_data_dir,
                                           'base/train.json'), lines=True)
    test_data = pd.read_json(os.path.join(output_data_dir,
                                          'base/test.json'), lines=True)
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train.json')), 'w') as fout:
        for _, record in train_data.iterrows():
            output_record = {
                'text': (record['clinical'] + " " + record['description'] + " " + record['diagnosis']).strip()
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')

    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': (record['clinical'] + " " + record['description'] + " " + record['diagnosis']).strip()
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')


def construct_t2t_data(output_data_dir, num_kfold):
    os.makedirs(os.path.join(output_data_dir, 't2t'), exist_ok=True)
    train_data = pd.read_json(os.path.join(output_data_dir,
                                           'base/train.json'), lines=True)
    test_data = pd.read_json(os.path.join(output_data_dir,
                                          'base/test.json'), lines=True)

    train_records, test_records = [], []

    for _, record in train_data.iterrows():
        record['text'] = (record['clinical'] + " " + record['description']).strip()
        record['summary'] = record['diagnosis']
        train_records.append(record.to_dict())

    for _, record in test_data.iterrows():
        record['text'] = (record['clinical'] + " " + record['description']).strip()
        record['summary'] = record['diagnosis']
        test_records.append(record.to_dict())

    # all_data
    all_train_records = [
        {'text': record['text'], 'summary': record['summary']}
        for record in train_records
    ]
    all_test_records = [
        {'text': record['text'], 'summary': record['summary']}
        for record in test_records
    ]
    pd.DataFrame(all_train_records).to_json(os.path.join(output_data_dir, 't2t/train.json'), orient='records', lines=True)
    pd.DataFrame(all_test_records).to_json(os.path.join(output_data_dir, 't2t/test.json'), orient='records',
                                           lines=True)
    os.makedirs(os.path.join(output_data_dir, f't2t/all'), exist_ok=True)
    pd.DataFrame(all_train_records).to_json(os.path.join(output_data_dir, 't2t/all/train.json'), orient='records', lines=True)
    pd.DataFrame(all_test_records).to_json(os.path.join(output_data_dir, 't2t/all/test.json'), orient='records',
                                           lines=True)

    for kfold in range(0, num_kfold):
        os.makedirs(os.path.join(output_data_dir, f't2t/{kfold}'), exist_ok=True)
        cur_train_records, cur_test_records = [], []
        for record in train_records:
            if record['kfold_id'] == kfold:
                cur_test_records.append({
                    'text': record['text'],
                    'summary': record['summary']
                })
            else:
                cur_train_records.append({
                    'text': record['text'],
                    'summary': record['summary']
                })
        pd.DataFrame(cur_train_records).to_json(os.path.join(output_data_dir, f't2t/{kfold}/train.json'), orient='records', lines=True)
        pd.DataFrame(cur_test_records).to_json(os.path.join(output_data_dir, f't2t/{kfold}/test.json'), orient='records',
                                               lines=True)


def construct_data(input_data_dir="raw_data",
                   output_data_dir="data",
                   stage=0,
                   num_kfold=5,
                   test_data_size=2000):
    # stage: 0 or 1.
    # 创建数据文件夹
    os.makedirs(output_data_dir, exist_ok=True)
    # 拆分训练集和测试集
    logger.info('拆分训练集合测试集')
    split_train_test_data(input_data_dir,
                          output_data_dir,
                          stage, num_kfold, test_data_size)
    # 构造clm训练数据
    logger.info('构造clm训练数据')
    construct_clm_data(output_data_dir)
    # 构造mlm训练
    logger.info('构造mlm训练数据')
    construct_mlm_data(output_data_dir)
    # 构建pt2t预训练
    logger.info('构建pt2t预训练数据')
    construct_pt2t_data(output_data_dir)
    # 构建t2t训练
    logger.info('构建t2t训练数据')
    construct_t2t_data(output_data_dir, num_kfold)




if __name__ == '__main__':
    fire.Fire(construct_data)