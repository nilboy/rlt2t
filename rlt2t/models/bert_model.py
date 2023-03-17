from typing import Any, List

from torchmetrics.regression.mse import MeanSquaredError
from transformers import AutoModelForSequenceClassification
from pytorch_lightning import LightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from torch.optim import AdamW


class BertModelForRegression(LightningModule):
    def __init__(self,
                 init_model: str,
                 lr: float=0.001,
                 weight_decay: float=0.0005,
                 warmup_step: int=1000,
                 max_iters: int=10000):
        super().__init__()
        self.save_hyperparameters()
        self.bert = AutoModelForSequenceClassification.from_pretrained(init_model,
                                                                       num_labels=1, return_dict=True)

        self.train_accuracy = MeanSquaredError()
        self.val_accuracy = MeanSquaredError()
        self.test_accuracy = MeanSquaredError()

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, labels=labels)
        return output.loss, output.logits

    def step(self, batch: Any):
        loss, logits = self.forward(batch['input_ids'],
                                    batch['token_type_ids'],
                                    batch['attention_mask'],
                                    batch['labels'])
        return loss, logits, batch['labels']

    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/mse', acc, on_step=True, on_epoch=True, prog_bar=True)
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
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/mse", acc, on_step=False, on_epoch=True)

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
