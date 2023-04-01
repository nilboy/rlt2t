# rlt2t

## vocab
```
0: [PAD]
1: [unused1]
2: [unused2]
3: [unused3]
... ...
100: [UNK]
101: [CLS]
102: [SEP]
103: [MASK]
104: <S>
105: <T>
```

## 训练
1. 构造数据
```
python tools/construct_data.py
```
2. 转换模型
```
python tasks/convert_models/convert_model.py --input_model_name=/root/autodl-tmp/Erlangshen-MegatronBert-1.3B \
 --output_model_name=/root/autodl-tmp/pmodels/Erlangshen-MegatronBert-1.3B \
 --model_type=bert \
 --vocab_size=2000
 
 python tasks/convert_models/convert_model.py --input_model_name=/root/autodl-tmp/Randeng-T5-784M \
 --output_model_name=/root/autodl-tmp/pmodels/Randeng-T5-784M \
 --model_type=t5 \
 --vocab_size=2000

python tasks/convert_models/convert_model.py --input_model_name=/root/autodl-tmp/Wenzhong2.0-GPT2-3.5B-chinese \
--output_model_name=/root/autodl-tmp/pmodels/Wenzhong2.0-GPT2-3.5B-chinese \
--model_type=gpt2 \
--vocab_size=2000
```
