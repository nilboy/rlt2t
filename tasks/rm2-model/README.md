# Finetune-regress-model
预训练mlm模型

# 准备数据

```
训练数据格式, jsonl:
{"text": "xxxxx", "summary": "xxxx", "label": 0.1}
{"text": "xxxxx", "summary": "xxxx", "label": 0.3}
```

# 训练

```
bash ./tasks/ft-regress-model/run.sh
```
