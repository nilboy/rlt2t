from typing import Any, List

import torch
from torchmetrics import BLEUScore
from transformers import AutoModelForSeq2SeqLM, BertTokenizer
from pytorch_lightning import LightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW


class T2TModel(LightningModule):
    def __init__(self,
                 init_model: str,
                 eos_id: int = 105,
                 lr: float=0.001,
                 weight_decay: float=0.0005,
                 warmup_step: int=1000,
                 max_iters: int=10000):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(init_model, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(init_model)
        self.tokenizer.bos_token_id = 0
        self.tokenizer.eos_token_id = eos_id
        self.val_bleu = BLEUScore(4)
        self.test_bleu = BLEUScore(4)
        self.eos_id = eos_id

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None):
        output = self.model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return output.loss, output.logits

    def step(self, batch: Any):
        loss, logits = self.forward(batch['input_ids'],
                                    batch['attention_mask'],
                                    batch['labels'])
        return loss, logits, batch['labels']

    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
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

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/bleu4", bleu4, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
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
