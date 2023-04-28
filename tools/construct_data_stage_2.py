import random

import fire
import os
import pandas as pd
import random
import json
from loguru import logger

split_token = '1799'


def split_train_test_data(input_data_dir,
                          output_data_dir,
                          test_data_size):
    os.makedirs(os.path.join(output_data_dir, 'base'), exist_ok=True)
    file_map = {
        'stage_1_train': {
            'filename': 'train.csv',
            'col_names': ['report_id', 'description', "diagnosis"]
        },
        'stage_2_train': {
            'filename': 'semi_train.csv',
            'col_names': ["report_id", "description", "diagnosis", "clinical"]
        },
        'stage_1_test_b': {
            'filename': 'preliminary_b_test.csv',
            'col_names': ['report_id', 'description']
        }
    }
    test_records = []
    all_records = []
    for file_type, file_info in file_map.items():
        df = pd.read_csv(os.path.join(input_data_dir, file_info['filename']),
                         header=None, index_col=False,
                         names=file_info['col_names'])
        records = df.to_dict('records')
        random.shuffle(records)
        if file_type == 'stage_2_train':
            test_records.extend(records[0:test_data_size])
            records = records[test_data_size:]
        for idx, record in enumerate(records):
            if not record.get('diagnosis', '') or str(record['diagnosis']) == 'nan':
                record['diagnosis'] = ''
            if not record.get('clinical', '') or str(record['clinical']) == 'nan':
                record['clinical'] = ''
            record['type'] = file_type
        all_records.extend(records)
    random.shuffle(all_records)

    for record in test_records:
        if not record.get('diagnosis', '') or str(record['diagnosis']) == 'nan':
            record['diagnosis'] = ''
        if not record.get('clinical', '') or str(record['clinical']) == 'nan':
            record['clinical'] = ''
        record['type'] = 'test'

    with open(os.path.join(output_data_dir, 'base/train.json'), 'w') as fout:
        for record in all_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    with open(os.path.join(output_data_dir, 'base/test.json'), 'w') as fout:
        for record in test_records:
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def construct_pt2t_data(output_data_dir):
    os.makedirs(os.path.join(output_data_dir, 'pt2t'), exist_ok=True)
    train_data = pd.read_json(os.path.join(output_data_dir,
                                           'base/train.json'), lines=True)
    test_data = pd.read_json(os.path.join(output_data_dir,
                                          'base/test.json'), lines=True)

    train_records = []
    for _, record in train_data.iterrows():
        output_record = {
            'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip(),
                              split_token, record['diagnosis'].strip()]),
            'type': record['type']
        }
        train_records.append(output_record)
    # stage_2_train
    os.makedirs(os.path.join(output_data_dir, 'pt2t/train_2'), exist_ok=True)
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train_2/train.json')), 'w') as fout:
        for record in train_records:
            if record['type'] in ['stage_2_train']:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train_2/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip(),
                                  split_token, record['diagnosis'].strip()]),
                'type': record['type']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    # stage_2_train + stage_1_train
    os.makedirs(os.path.join(output_data_dir, 'pt2t/train_2_train'), exist_ok=True)
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train_2_train/train.json')), 'w') as fout:
        for record in train_records:
            if record['type'] in ['stage_2_train', 'stage_1_train']:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train_2_train/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip(),
                                  split_token, record['diagnosis'].strip()]),
                'type': record['type']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    # stage_2_train + stage_1_train + stage_1_test
    os.makedirs(os.path.join(output_data_dir, 'pt2t/train_2_train_test'), exist_ok=True)
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train_2_train_test/train.json')), 'w') as fout:
        for record in train_records:
            if record['type'] in ['stage_2_train', 'stage_1_train', 'stage_1_test_b']:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    # test.
    with open(os.path.join(os.path.join(output_data_dir, 'pt2t/train_2_train_test/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip(),
                                  split_token, record['diagnosis'].strip()]),
                'type': record['type']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')



def construct_t2t_data(output_data_dir):
    os.makedirs(os.path.join(output_data_dir, 't2t'), exist_ok=True)
    train_data = pd.read_json(os.path.join(output_data_dir,
                                           'base/train.json'), lines=True)
    test_data = pd.read_json(os.path.join(output_data_dir,
                                          'base/test.json'), lines=True)

    train_records = []
    for _, record in train_data.iterrows():
        output_record = {
            'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip()]),
            'summary': record['diagnosis'].strip(),
            'type': record['type']
        }
        train_records.append(output_record)
    # stage_2_train
    os.makedirs(os.path.join(output_data_dir, 't2t/train_2'), exist_ok=True)
    with open(os.path.join(os.path.join(output_data_dir, 't2t/train_2/train.json')), 'w') as fout:
        for record in train_records:
            if record['type'] in ['stage_2_train']:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    # test.
    with open(os.path.join(os.path.join(output_data_dir, 't2t/train_2/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip()]),
                'summary': record['diagnosis'].strip(),
                'type': record['type']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    # stage_2_train + stage_1_train
    os.makedirs(os.path.join(output_data_dir, 't2t/train_2_train'), exist_ok=True)
    with open(os.path.join(os.path.join(output_data_dir, 't2t/train_2_train/train.json')), 'w') as fout:
        for record in train_records:
            if record['type'] in ['stage_2_train', 'stage_1_train']:
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    # test.
    with open(os.path.join(os.path.join(output_data_dir, 't2t/train_2_train/test.json')), 'w') as fout:
        for _, record in test_data.iterrows():
            output_record = {
                'text': " ".join([record['clinical'].strip(), split_token, record['description'].strip()]),
                'summary': record['diagnosis'].strip(),
                'type': record['type']
            }
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')


def construct_data(input_data_dir="raw_data",
                   output_data_dir="data",
                   test_data_size=1000):
    # stage: 0 or 1.
    # 创建数据文件夹
    os.makedirs(output_data_dir, exist_ok=True)
    # 拆分训练集和测试集
    logger.info('拆分训练集合测试集')
    split_train_test_data(input_data_dir,
                          output_data_dir,
                          test_data_size)
    # 构建pt2t预训练
    logger.info('构建pt2t预训练数据')
    construct_pt2t_data(output_data_dir)
    # 构建t2t训练
    logger.info('构建t2t训练数据')
    construct_t2t_data(output_data_dir)


if __name__ == '__main__':
    fire.Fire(construct_data)