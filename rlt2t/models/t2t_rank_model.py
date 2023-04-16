from typing import Any, List

import torch
from torchmetrics import BLEUScore
from transformers import AutoModelForSeq2SeqLM, BertTokenizer
from pytorch_lightning import LightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from rlt2t.utils.evaluate import calculate_scores


class T2TRankModel(LightningModule):
    def __init__(self,
                 init_model: str,
                 eos_id: int = 105,
                 lr: float=0.001,
                 weight_decay: float=0.0005,
                 warmup_step: int=1000,
                 max_iters: int=10000,
                 rank_start_iters: int = 500,
                 rank_loss_rate: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(init_model, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(init_model)
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = eos_id
        self.val_bleu = BLEUScore(4)
        self.test_bleu = BLEUScore(4)
        self.eos_id = eos_id
        self.rank_start_iters = rank_start_iters
        self.rank_loss_rate = rank_loss_rate

    def get_score(self, logits, labels):
        # [batch_size, ]
        bs, seq_len, vocab_size = logits.shape
        score = -CrossEntropyLoss(reduction="none")(logits.view(-1, vocab_size),
                                                    labels.view(-1)).reshape((bs, seq_len))
        score = score.sum(axis=-1) / torch.sum(labels != -100, axis=-1)
        return score

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                rank_a_labels=None,
                rank_b_labels=None,
                orig_input_ids=None,
                orig_attention_mask=None,
                use_rank=None):
        output = self.model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        ft_loss, rank_loss = output.loss, 0.0
        rank_acc = 0.0
        if self.training and self.trainer.global_step >= self.rank_start_iters:
            output_rank_a = self.model(orig_input_ids,
                                       attention_mask=orig_attention_mask,
                                       labels=rank_a_labels)
            output_rank_b = self.model(orig_input_ids,
                                       attention_mask=orig_attention_mask,
                                       labels=rank_b_labels)
            rank_a_score = self.get_score(output_rank_a.logits,
                                          rank_a_labels)
            rank_b_score = self.get_score(output_rank_b.logits,
                                          rank_b_labels)
            rank_loss = torch.maximum(torch.zeros_like(rank_a_score),
                                      rank_b_score - rank_a_score)
            rank_loss = torch.mean(rank_loss * use_rank)
            rank_acc = torch.mean(rank_a_score >= rank_b_score, dtype=rank_loss.dtype)

        return ft_loss, rank_loss, rank_acc, output.logits

    def step(self, batch: Any):
        ft_loss, rank_loss, rank_acc, logits = self.forward(batch['input_ids'],
                                                  batch['attention_mask'],
                                                  batch['labels'],
                                                  batch['rank_a_labels'],
                                                  batch['rank_b_labels'],
                                                  batch['orig_input_ids'],
                                                  batch['orig_attention_mask'],
                                                  batch['use_rank'])
        return ft_loss, rank_loss, rank_acc, logits, batch['labels']

    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        ft_loss, rank_loss, rank_acc, preds, targets = self.step(batch)
        if self.rank_loss_rate > 0.0:
            loss = ft_loss + rank_loss * self.rank_loss_rate
        else:
            loss = ft_loss
        #loss = rank_loss * self.rank_loss_rate
        self.log('train/ft_loss', ft_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/rank_loss', rank_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/rank_acc', rank_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        ft_loss, rank_loss, rank_acc, preds, targets = self.step(batch)
        loss = ft_loss + rank_loss * self.rank_loss_rate
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
            output_text = " ".join([str(item) for item in filtered_tids])
            output_texts.append(output_text)
        label_texts = []
        labels = batch['labels'].detach().cpu().tolist()
        for i in range(len(labels)):
            filtered_tids = []
            tids = labels[i]
            for tid in tids:
                if tid > 0:
                    filtered_tids.append(tid)
            output_text = " ".join([str(item) for item in filtered_tids])
            label_texts.append(output_text)

        bleu4 = self.val_bleu(output_texts,
                      [[item] for item in label_texts])

        self.log('val/ft_loss', ft_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/rank_loss', rank_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/rank_acc', rank_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/bleu4", bleu4, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "e_label_texts": label_texts, "e_output_texts": output_texts}

    def validation_epoch_end(self, outputs: List[Any]):
        preds, labels = [], []
        for item in outputs:
            preds.extend(item['e_output_texts'])
            labels.extend(item['e_label_texts'])
        m_score = calculate_scores(labels, preds, bleu4_rate=0.0)
        self.log("val/m_score", m_score.mean(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        ft_loss, rank_loss, rank_acc, preds, targets = self.step(batch)
        loss = ft_loss + rank_loss * self.rank_loss_rate
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
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                              num_warmup_steps=self.hparams.warmup_step,
                                                              num_training_steps=self.hparams.max_iters)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
