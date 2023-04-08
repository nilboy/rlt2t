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
3. 预训练clm、mlm、t2t模型
```
bash tasks/pretrain-mlm/run.sh ddp 1 11889
bash tasks/pretrain-t2t/run.sh ddp 1 12889
bash tasks/train-clm/run.sh ddp 1 13889
```
4. 训练kfold, sft-t2t模型
```
bash tasks/sft-t2t/run.sh ddp 1 10889 0
bash tasks/sft-t2t/run.sh ddp 1 11889 1
bash tasks/sft-t2t/run.sh ddp 1 12889 2
bash tasks/sft-t2t/run.sh ddp 1 13889 3
bash tasks/sft-t2t/run.sh ddp 1 14889 4
# 导出模型
# ct2-transformers-converter --model facebook/m2m100_418M --output_dir ct2_model
```
5. 通过clm模型构造大数据集
```
# 导出模型
ct2-transformers-converter --model facebook/m2m100_418M --output_dir ct2_model
# 生成数据
python tools/generate_clm_data.py --model_path clm_ct2 --output_file data/gen_clm.json --generate_num 1000000
```
6. 构造RM数据集
```
python construct_rm_data.py
```
7. 训练RM模型
```
bash ./ft-regress-model/run.sh
```
8. 构造弱标签数据集
```
python generate_weak_data.py
```
9. 训练弱标签t2t模型
10. 训练人工标签t2t模型