# Finetune-T2T
finetune text2text 模型.

# 准备数据

```
训练数据格式, jsonl:
{"text": "xxxxx", "summary": "xxxx"}
{"text": "xxxxx", "summary": "xxxx"}
```

# 训练

```
bash ./tasks/sft-t2t/run.sh
```
