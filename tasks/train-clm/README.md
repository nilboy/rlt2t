# Train-CLM
预训练或微调CLM模型

# 准备数据

```
训练数据格式, json:(记录中可以有其他属性)
{"text": "xxxxxxxxxxxxx"}
{"text": "xxxxxxxxxxxxxxxxxxx"}
```

# 训练

```
bash ./tasks/train-clm/run.sh ddp 2
or  bash ./tasks/train-clm/run.sh deepspeed
```

# 模型转换

```
ct2-transformers-converter --model input_model_dir --output_dir output_model_dir
```
