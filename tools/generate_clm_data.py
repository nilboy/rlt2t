import fire
from rlt2t.engines.clm_engine import CLMEngineCT2
from tqdm.auto import tqdm
import json
import os

def generate_records(model_path,
                     output_file,
                     generate_num,
                     batch_size=64,
                     min_length=8,
                     max_length=164,
                     sampling_topk=10,
                     sampling_temperature=0.8):
    engine = CLMEngineCT2(model_path,
                          compute_type='int8', num_words=1800)
    texts = set()
    pre_texts = []
    if os.path.exists(output_file):
        for line in open(output_file):
            text = json.loads(line)['text']
            pre_texts.append(text)
            texts.add(text)

    with open(output_file, 'w') as fout:
        for pre_text in pre_texts:
            item = {'text': pre_text}
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        batch_num = generate_num // batch_size
        for i in tqdm(range(batch_num), 'generate...'):
            results = engine.predict_records(batch_size,
                                             min_length=min_length,
                                             max_length=max_length,
                                             sampling_topk=sampling_topk,
                                             sampling_temperature=sampling_temperature)
            for item in results:
                if item not in texts:
                    texts.add(item)
                    item = {'text': item}
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    fire.Fire(generate_records)