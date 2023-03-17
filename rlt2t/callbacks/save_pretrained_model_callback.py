from pytorch_lightning.callbacks import ModelCheckpoint
import os

class SavePretrainedModelCallback(ModelCheckpoint):
    def _save_checkpoint(self, trainer, filepath: str) -> None:
        super(SavePretrainedModelCallback, self)._save_checkpoint(trainer, filepath)
        trainer.model.module.module.bert.save_pretrained(os.path.join(filepath, 'bert'))
