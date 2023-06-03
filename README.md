# rlt2t

Reinforcement learning text to text.

## 算法

### 整体思路介绍
整体架构是多模型生成多个候选答案，Rerank模型排序，选取最好的答案作为最终的输出。

生成模型和Rerank模型都选择bart结构。 生成模型 和 Rerank模型使用同一个模型，用RL训练:

强化学习: 上一个epoch保存的模型作为策略生成模型，生成a,b两个答案，通过评测指标计算score_a, score_b。Rerank模型对a,b进行比较。计算rank_loss。

rank_loss 和 自回归loss 加权作为最终loss

### Rank loss 计算
```
def get_rank_outputs(self, orig_input_ids=None,
                     orig_attention_mask=None,
                     rank_a_labels=None,
                     rank_b_labels=None,
                     use_rank=None):
    output_rank_a = self.model(orig_input_ids,
                               attention_mask=orig_attention_mask,
                               labels=rank_a_labels)
    output_rank_b = self.model(orig_input_ids,
                               attention_mask=orig_attention_mask,
                               labels=rank_b_labels)
    rank_a_labels_pad = rank_a_labels.masked_fill(rank_a_labels == -100, 0)
    rank_b_labels_pad = rank_b_labels.masked_fill(rank_b_labels == -100, 0)
    # (batch_size, seq_len)
    rank_a_logits = torch.gather(output_rank_a.logits, 2, rank_a_labels_pad.unsqueeze(2)).squeeze(-1)
    rank_b_logits = torch.gather(output_rank_b.logits, 2, rank_b_labels_pad.unsqueeze(2)).squeeze(-1)
    diff_logits = rank_a_logits - rank_b_logits
    rank_loss = -torch.log(torch.sigmoid(diff_logits))
    mask_rate = self.build_mask_rate(rank_a_labels, rank_b_labels, rank_a_logits.dtype, alpha=self.delay_alpha)
    rank_loss = rank_loss * mask_rate

    # build select mask
    select_mask = (rank_a_labels != -100) & (rank_b_labels != -100) & (mask_rate > 1e-8)
    select_mask = select_mask & (use_rank.unsqueeze(-1).to(select_mask.dtype))

    rank_loss = torch.masked_select(rank_loss, select_mask).sum() / (torch.masked_select(mask_rate, select_mask).sum() + 1e-8)
    rank_acc = torch.sum(diff_logits > 0) / diff_logits.shape[0]
    return rank_loss, rank_acc

def build_mask_rate(self, rank_a_labels, rank_b_labels,
                    dtype,
                    alpha=0.9):
    batch_size, seq_len = rank_a_labels.shape
    mask_rate = torch.zeros_like(rank_a_labels, dtype=dtype)
    for i in range(batch_size):
        items = (rank_a_labels[i] != rank_b_labels[i]).nonzero()
        start_idx = items[0].item() if len(items) > 0 else seq_len
        mask_rate[i, start_idx:] = alpha ** torch.arange(seq_len - start_idx, device=rank_a_labels.device)
    return mask_rate
```

## 训练流程
* 处理训练数据
```
python tools/construct_data_stage_2.py
```
* 转换模型
```
python tasks/convert_models/convert_model.py \
 --input_model_name=pretrain_models/uer_bart_base \
 --output_model_name=pmodels/uer_bart_base \
 --model_type=bart \
 --vocab_size=2000

```
* DAE预训练
```
python tasks/pl-pretrain-t2t/run.py fit \
    --config tasks/pl-pretrain-t2t/config_uer_bart_base.yaml
```
* 训练生成模型
```
python tasks/pl-sft-t2t/run.py fit --config tasks/pl-sft-t2t/base/uer_bart_base.yaml
```
* 训练rank模型
用上面训练的生成模型作为初始化，加入rank_loss继续微调
```
# 启动策略生成服务

mkdir -p output-models/uer_bart_base-rank
cp -r output-models/uer_bart_base/last.ckpt.dir output-models/uer_bart_base-rank/epoch_0.ckpt.dir
python tools/rank_data_app.py output-models/uer_bart_base-rank 9550
python tasks/pl-sft-t2t/run.py fit --config tasks/pl-sft-t2t/rank/uer_bart_base.yaml
```
* 导出模型，转化为ctranslat2格式
```
ct2-transformers-converter --model output-models/uer_bart_base-rank --output_dir sub-models/uer_bart_base_rank --quantization int8
```

## 推理流程
```
python index.py
```