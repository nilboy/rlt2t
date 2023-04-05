import sys
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import LightningDataModule, LightningModule
sys.path.insert(0, '.')

def cli_main():
    cli = LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True,
                       parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    cli_main()