from typing import Any, List

from torchmetrics.regression.mse import MeanSquaredError
from transformers import AutoModelForSequenceClassification
from pytorch_lightning import LightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW
from transformers import BertTokenizer
import torch


class BertModelRMForRegression(LightningModule):
    def __init__(self,
                 init_model: str,
                 lr: float=0.001,
                 weight_decay: float=0.0005,
                 warmup_step: int=1000,
                 max_iters: int=10000):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(init_model,
                                                                       num_labels=1, return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained(init_model)

        self.train_accuracy = MeanSquaredError()
        self.val_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()

    def forward(self,
                input_ids_a=None,
                token_type_ids_a=None,
                attention_mask_a=None,
                input_ids_b=None,
                token_type_ids_b=None,
                attention_mask_b=None):
        output_a = self.model(input_ids_a, attention_mask=attention_mask_a,
                              token_type_ids=token_type_ids_a)
        output_b = self.model(input_ids_b, attention_mask=attention_mask_b,
                              token_type_ids=token_type_ids_b)
        return output_a.logits, output_b.logits

    def step(self, batch: Any):
        logits_a, logits_b = self.forward(batch['input_ids_a'],
                                          batch['token_type_ids_a'],
                                          batch['attention_mask_a'],
                                          batch['input_ids_b'],
                                          batch['token_type_ids_b'],
                                          batch['attention_mask_b'])
        diff_score = batch['score_a'] - batch['score_b']
        loss = -torch.log(torch.sigmoid(logits_a - logits_b)) * diff_score
        loss = loss.sum() / (diff_score.sum() + 1e-8)
        acc = torch.where(logits_a > logits_b, batch['score_a'], batch['score_b']).mean()
        return loss, acc

    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        loss, acc = self.step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {
            'loss': loss
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, acc = self.step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

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
