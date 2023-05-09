from typing import Any, List

import torch
from torchmetrics import BLEUScore
from transformers import AutoModelForSeq2SeqLM, BertTokenizer, AutoConfig
from pytorch_lightning import LightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from rlt2t.utils.evaluate import calculate_scores
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers import Adafactor
import json

import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='embed_tokens.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embed_tokens.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr=0.0,
                                    last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(~pad_mask[..., None], 0.)
        q_loss.masked_fill_(~pad_mask[..., None], 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum() / (pad_mask.sum() + 1e-8)
    q_loss = q_loss.sum() / (pad_mask.sum() + 1e-8)

    loss = (p_loss + q_loss) / 2
    return loss


class T2TRankModel(LightningModule):
    def __init__(self,
                 init_model: str,
                 eos_id: int = 105,
                 lr: float=0.001,
                 weight_decay: float=0.0005,
                 warmup_step: int=1000,
                 max_iters: int=10000,
                 min_lr: float = 1e-6,
                 dropout_rate: float = 0.1,
                 rank_start_iters: int = 500,
                 rank_loss_rate: float = 1.0,
                 delay_alpha: float = 0.9,
                 rdrop_alpha: float = 0.0,
                 rdrop_start_steps: int = 500,
                 use_fgm: bool = False,
                 fgm_epsilon: float = 0.1,
                 fgm_start_steps: int = 500):
        super().__init__()
        self.save_hyperparameters()
        config = AutoConfig.from_pretrained(init_model)
        config.dropout = dropout_rate
        config.return_dict = True
        self.model = AutoModelForSeq2SeqLM.from_pretrained(init_model, config=config)
        self.tokenizer = BertTokenizer.from_pretrained(init_model)
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = eos_id
        self.val_bleu = BLEUScore(4)
        self.test_bleu = BLEUScore(4)
        self.eos_id = eos_id
        self.rank_start_iters = rank_start_iters
        self.rank_loss_rate = rank_loss_rate
        self.delay_alpha = delay_alpha
        self.rdrop_alpha = rdrop_alpha
        self.rdrop_start_steps = rdrop_start_steps
        self.min_lr = min_lr
        self.use_fgm = use_fgm
        self.fgm_epsilon = fgm_epsilon
        self.fgm_start_steps = fgm_start_steps
        if self.use_fgm:
            self.automatic_optimization = False
            self.fgm = FGM(self)

    def get_score(self, logits, labels):
        # [batch_size, ]
        bs, seq_len, vocab_size = logits.shape
        score = -CrossEntropyLoss(reduction="none")(logits.view(-1, vocab_size),
                                                    labels.view(-1)).reshape((bs, seq_len))
        score = score.sum(axis=-1) / torch.sum(labels != -100, axis=-1)
        return score

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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None):
        output = self.model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return output

    def step(self, batch: Any):
        output = self.forward(batch['input_ids'],
                              batch['attention_mask'],
                              batch['labels'])
        return output

    def get_predict_model(self):
        swa_callback = None
        for item in self.trainer.callbacks:
            if isinstance(item, StochasticWeightAveraging):
                swa_callback = item
                break
        if swa_callback is None or swa_callback._average_model.model.device != self.model.device:
            return self.model
        else:
            return swa_callback._average_model.model

    def get_loss(self, batch: Any):
        loss = 0
        if self.rdrop_alpha > 0.0 and self.trainer.global_step >= self.rdrop_start_steps:
            output1 = self.forward(batch['input_ids'],
                                   batch['attention_mask'],
                                   labels=batch['labels'])
            output2 = self.forward(batch['input_ids'],
                                   batch['attention_mask'],
                                   labels=batch['labels'])
            ce_loss = 0.5 * (output1.loss + output2.loss)
            self.log('train/ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=False)
            kl_loss = compute_kl_loss(output1.logits, output2.logits, batch['labels'] != -100)
            self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=False)
            loss += ce_loss + self.rdrop_alpha * kl_loss
        else:
            output = self.forward(batch['input_ids'],
                                  batch['attention_mask'],
                                  labels=batch['labels'])
            ce_loss = output.loss
            self.log('train/ce_loss', ce_loss, on_step=True, on_epoch=True, prog_bar=False)
            loss += ce_loss

        if self.training and self.rank_loss_rate > 0.0 and self.trainer.global_step >= self.rank_start_iters:
            rank_loss, rank_acc = self.get_rank_outputs(orig_input_ids=batch['orig_input_ids'],
                                                        orig_attention_mask=batch['orig_attention_mask'],
                                                        rank_a_labels=batch['rank_a_labels'],
                                                        rank_b_labels=batch['rank_b_labels'],
                                                        use_rank=batch['use_rank'])
            self.log('train/rank_loss', rank_loss, on_step=True, on_epoch=True, prog_bar=False)
            loss += rank_loss * self.rank_loss_rate
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss


    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        if not self.use_fgm:
            loss = self.get_loss(batch)
            self.log('monitoring_step', self.global_step)
            return {
                'loss': loss
            }
        elif self.trainer.global_step < self.fgm_start_steps:
            sch = self.lr_schedulers()
            sch.step()
            opt = self.optimizers()
            opt.zero_grad()
            loss = self.get_loss(batch)
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            return {
                'loss': loss
            }
        else:
            sch = self.lr_schedulers()
            sch.step()
            opt = self.optimizers()
            opt.zero_grad()
            loss = self.get_loss(batch)
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.fgm.attack()
            loss_adv = self.get_loss(batch)
            self.fgm.restore()
            self.manual_backward(loss_adv)
            opt.step()
            return {
                'loss': loss
            }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        output = self.forward(batch['input_ids'],
                              batch['attention_mask'],
                              labels=batch['labels'])
        loss = output.loss
        with torch.no_grad():
            outputs = self.get_predict_model().generate(input_ids=batch['input_ids'],
                                                        attention_mask=batch['attention_mask'],
                                                        num_beams=2,
                                                        max_length=96)
        outputs = outputs.detach().cpu().tolist()
        output_texts = []
        for i in range(len(outputs)):
            filtered_tids = []
            tids = outputs[i]
            for tid in tids:
                if tid == self.eos_id:
                    break
                if tid > 0:
                    filtered_tids.append(tid)
            output_text = " ".join([str(item) for item in filtered_tids])
            output_texts.append(output_text)
        label_texts = []
        labels = batch['labels'].detach().cpu().tolist()
        for i in range(len(labels)):
            filtered_tids = []
            tids = labels[i]
            for tid in tids:
                if tid == self.eos_id:
                    break
                if tid > 0:
                    filtered_tids.append(tid)
            output_text = " ".join([str(item) for item in filtered_tids])
            label_texts.append(output_text)

        bleu4 = self.val_bleu(output_texts,
                      [[item] for item in label_texts])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/bleu4", bleu4, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "e_label_texts": label_texts, "e_output_texts": output_texts}

    def validation_epoch_end(self, outputs: List[Any]):
        preds, labels = [], []
        for item in outputs:
            preds.extend(item['e_output_texts'])
            labels.extend(item['e_label_texts'])
        # with open('text.json', 'w') as fout:
        #     for pred, label in zip(preds, labels):
        #         fout.write(json.dumps({'pred': pred, 'label': label}, ensure_ascii=False) + '\n')
        m_score = calculate_scores(labels, preds, bleu4_rate=0.0)
        self.log("val/m_score", m_score.mean(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        output = self.forward(batch['input_ids'],
                              batch['attention_mask'],
                              labels=batch['labels'])
        loss = output.loss
        with torch.no_grad():
            outputs = self.model.generate(input_ids=batch['input_ids'],
                                          attention_mask=batch['attention_mask'],
                                          num_beams=2,
                                          max_length=96)
        outputs = outputs.detach().cpu().tolist()
        output_texts = []
        for i in range(len(outputs)):
            filtered_tids = []
            tids = outputs[i]
            for tid in tids:
                if tid == self.eos_id:
                    break
                if tid > 0:
                    filtered_tids.append(tid)
            output_text = " ".join(filtered_tids)
            output_texts.append(output_text)
        label_texts = []
        labels = batch['labels'].detach().cpu().tolist()
        for i in range(len(labels)):
            filtered_tids = []
            tids = labels[i]
            for tid in tids:
                if tid > 0:
                    filtered_tids.append(tid)
            output_text = " ".join(filtered_tids)
            label_texts.append(output_text)

        bleu4 = self.val_bleu(output_texts,
                      [[item] for item in label_texts])

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/bleu4", bleu4, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self) -> Any:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # not use adafactor.
        if isinstance(self.model, (T5ForConditionalGeneration, MT5ForConditionalGeneration)):
            optimizer = Adafactor(optimizer_grouped_parameters,
                                  lr=self.hparams.lr,
                                  eps=(1e-30, 1e-3),
                                  clip_threshold=1.0,
                                  decay_rate=-0.8,
                                  beta1=None,
                                  weight_decay=0.0,
                                  relative_step=False,
                                  scale_parameter=False,
                                  warmup_init=False)
        else:
            optimizer = AdamW(optimizer_grouped_parameters,
                              lr=self.hparams.lr,
                              weight_decay=self.hparams.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=self.hparams.warmup_step,
                                                    num_training_steps=self.hparams.max_iters,
                                                    min_lr=self.hparams.min_lr/self.hparams.lr)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
