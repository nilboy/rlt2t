import time
import json
from multiprocessing import Process, Queue

from flask import Flask, request, Response
from loguru import logger
from rlt2t.predictor.predictor import Predictor
import random
from rlt2t.utils.evaluate import calculate_scores
import sys

def producer(queue):
    t2t_model_paths = [
        "/root/autodl-tmp/ct2/0",
        "/root/autodl-tmp/ct2/1",
        "/root/autodl-tmp/ct2/2",
        "/root/autodl-tmp/ct2/3",
        "/root/autodl-tmp/ct2/4"
    ]
    rm_model_paths = []
    data_file = "/root/project/rlt2t/data/t2t/train.json"
    input_records = []
    for line in open(data_file):
        input_records.append(json.loads(line))
    while True:
        # 生成记录
        predictor = Predictor(random.sample(t2t_model_paths, 2),
                              rm_model_paths, num_hypotheses=4, sample_size=2)
        output_records_list = predictor.predict([item['text'] for item in input_records],
                                                return_best=False, use_beam_search=True, use_sample=True)

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
        # 尝试将记录放入队列
        for record_item in filtered_records:
            while True:
                if queue.full():
                    # 队列已满，等待队列被消耗一部分后继续生成
                    time.sleep(1)
                else:
                    # 将记录放入队列
                    queue.put(record_item)
                    break

app = Flask(__name__)

@app.route('/get_record', methods=['POST'])
def predict():
    record = None
    for i in range(3):
        if app.data_queue.empty():
            time.sleep(1)
        else:
            record = app.data_queue.get()
            break
    if record:
        return Response(json.dumps(record, ensure_ascii=False, indent=4), status=200)
    else:
        raise ValueError


if __name__ == '__main__':
    logger.info("rm data 服务... ...")
    queue = Queue(maxsize=100000)
    producer_process = Process(target=producer, args=(queue,))
    producer_process.start()
    app.data_queue = queue
    app.run(host="0.0.0.0", port=int(sys.argv[1]), debug=True, use_reloader=False)