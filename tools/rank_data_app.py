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


def get_last_model_path(base_model_dir):
    max_model_number, max_model_name = -1, ''
    if not os.path.exists(base_model_dir):
        return max_model_name
    for filename in os.listdir(base_model_dir):
        matched = re.match(r'epoch_(\d+)\.ckpt.dir', filename)
        if matched:
            model_steps = int(matched.groups()[0])
            if model_steps > max_model_number:
                max_model_number = model_steps
                max_model_name = filename
    return max_model_name


class Model(object):
    def __init__(self, base_model_dir,
                 file_path: str = "data/t2t/train.json",
                 interval_seconds: int = 10):
        self.file_path = file_path
        self.base_model_dir = base_model_dir
        self.ct2_model_dir = os.path.join(base_model_dir, 'ct2')
        self.interval_seconds = interval_seconds
        self.data = {}
        self.records = []
        self.max_model_name = ""
        # 定时调用线程
        self._rwlock = rwlock.RWLockFairD()
        self.init_data_from_file()
        logger.info(f"更新频率: {interval_seconds} s")
        if interval_seconds > 0:
            self._update_thread = threading.Thread(
                target=self._schedule_update, args=(interval_seconds,)
            )
            self._update_thread.start()
        else:
            self._update_thread = None

    def _schedule_update(self, interval_seconds: int) -> None:
        """定时调用_update函数"""
        while True:
            try:
                time.sleep(interval_seconds)
                self._update()
            except Exception as e:
                logger.warning(f"更新异常!" + traceback.format_exc())

    def init_data_from_file(self):
        gen_data = {}
        self.records = []
        with open(self.file_path) as fin:
            for line in fin:
                record = json.loads(line)
                self.records.append(record)
                gen_data[record['text']] = {
                    'text': record['text'],
                    'summary': record['summary'],
                    'rank_a': record['summary'],
                    'rank_b': record['summary'],
                    'use_rank': 0.0
                }
        with self._rwlock.gen_wlock():
            self.data = gen_data

    def _update(self) -> None:
        beam_size = 6
        cur_max_model_name = get_last_model_path(self.base_model_dir)
        if cur_max_model_name == self.max_model_name or cur_max_model_name == "":
            return
        self.max_model_name = cur_max_model_name
        cur_model_path = os.path.join(self.base_model_dir, self.max_model_name)
        ct2_cmd = f'ct2-transformers-converter --model={cur_model_path} --output_dir={self.ct2_model_dir} --force'
        logger.info(ct2_cmd)
        os.system(ct2_cmd)
        engine = T2TEngineCT2(self.ct2_model_dir, compute_type="int8", num_words=1800)
        texts = [item['text'] for item in self.records]
        summaries = [item['summary'] for item in self.records]
        output_texts_gen = engine.predict_records(texts, batch_size=64,
                                                  beam_size=1,
                                                  max_input_length=256,
                                                  max_decoding_length=96,
                                                  length_penalty=1.0,
                                                  num_hypotheses=beam_size,
                                                  sampling_topk=3,
                                                  sampling_temperature=0.8)
        # output_texts_gen = engine.predict_records(texts, batch_size=64,
        #                                           beam_size=4,
        #                                           max_input_length=256,
        #                                           max_decoding_length=96,
        #                                           length_penalty=1.0,
        #                                           num_hypotheses=1,
        #                                           sampling_topk=3,
        #                                           sampling_temperature=0.8)
        output_texts = []
        for i in range(len(self.records)):
            c_texts = list(set(output_texts_gen[i] + [self.records[i]['summary']]))
            if len(c_texts) < 2:
                c_texts.append(c_texts[0])
            output_texts.append(c_texts)
        output_texts_1, output_texts_2 = [], []
        for item in output_texts:
            idx1, idx2 = random.sample(range(0, len(item)), 2)
            output_texts_1.append(item[idx1])
            output_texts_2.append(item[idx2])
        output_texts_1_scores = calculate_scores(summaries, output_texts_1)
        output_texts_2_scores = calculate_scores(summaries, output_texts_2)
        gen_data = {}
        for i in range(len(self.records)):
            text = self.records[i]['text']
            summary = self.records[i]['summary']
            use_rank = 1.0
            if output_texts_1_scores[i] > output_texts_2_scores[i]:
                rank_a = output_texts_1[i]
                rank_b = output_texts_2[i]
                rank_a_score = output_texts_1_scores[i]
                rank_b_score = output_texts_2_scores[i]
            else:
                rank_a = output_texts_2[i]
                rank_b = output_texts_1[i]
                rank_a_score = output_texts_2_scores[i]
                rank_b_score = output_texts_1_scores[i]
            if rank_a == rank_b or rank_a_score == rank_b_score:
                use_rank = 0.0
            gen_data[text] = {
                'text': text,
                'summary': summary,
                'rank_a': rank_a,
                'rank_b': rank_b,
                'rank_a_score': rank_a_score,
                'rank_b_score': rank_b_score,
                'use_rank': use_rank
            }
        logger.info('begin update data...')
        with self._rwlock.gen_wlock():
            self.data = gen_data
        logger.info('update data finished...')

    def get_example(self, text: str):
        with self._rwlock.gen_rlock():
            example = self.data[text]
        return example


app = Flask(__name__)

app.model = Model(sys.argv[1])


@app.route('/get_rank_data', methods=['POST'])
def predict():
    text = request.json['text']
    example = app.model.get_example(text)
    return Response(json.dumps(example, ensure_ascii=False, indent=4), status=200)


if __name__ == '__main__':
    logger.info("rank data 服务... ...")
    app.run(host="0.0.0.0", port=int(sys.argv[2]), debug=True, use_reloader=False)
