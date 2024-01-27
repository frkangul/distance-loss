"""
This is the main script for training the image segmentation model. It includes the following steps:
1. Import necessary libraries
2. Define the main pipeline function that sets up the configuration, seeds, logger, model, callbacks, and trainer.
3. Start the training process and finally test the model.

Auther: Furkan Gul
Date: 24.08.2023
"""
# from torchmetrics import Dice # https://github.com/Lightning-AI/metrics/tree/master/src/torchmetrics
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

import warnings
from omegaconf import DictConfig, OmegaConf
import hydra

from src.models.plModel import ImageSegModel
from src.data.transforms import get_transforms
from src.utils import seed_everything, setup_wandb_and_logger, setup_pl_callbacks
warnings.filterwarnings("ignore") # category=DeprecationWarning


@hydra.main(version_base=None, config_path="./config", config_name="config")
def test_pipeline(cfg: DictConfig):
    """
    The main pipeline function that sets up the configuration, seeds, logger, model, callbacks, and trainer.
    It then starts the training process and finally tests the model.
    """
    pprint(OmegaConf.to_yaml(cfg)) # OmegaConf.to_yaml(cfg)

    seed_everything(cfg.exp.SEED)

    # wandb_logger = setup_wandb_and_logger(cfg)
    import os
    import wandb
    from dotenv import load_dotenv

    load_dotenv()
    WANDB_KEY= os.environ["WANDB"]
    wandb.login(key=WANDB_KEY)
    run = wandb.init()

    transforms = get_transforms(cfg)

    model = ImageSegModel(cfg, transforms["train"], transforms["val"], transforms["test"], transforms["unnorm"])
    ModelSummary(model) #to see detailed layer based parameter nums max_depth=-1

    callbacks = setup_pl_callbacks(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epoch,
        max_time=cfg.trainer.max_time, # Stop after max_time hours of training or when reaching max_epochs epochs
        precision=cfg.server.precision, # Mixed Precision (16-bit) combines the use of both 32 and 16-bit floating points to reduce memory footprint
        accelerator=cfg.server.accelerator, # gpu, tpu, auto
        devices=cfg.server.device_num,
        callbacks= callbacks,
        # logger=[CSVLogger(save_dir="logs/"), wandb_logger], # multiple loggers
        strategy=None if cfg.server.device_num==1 else 'dp', # 'dp' strategy is used when multiple-gpus with 1 machine.
        deterministic=cfg.exp.deterministic, # for reproducibity. 'warn' to use deterministic algorithms whenever possible
    )
    
    # wandb_logger.watch(model, log="all") # log and monitor gradients, parameter histogram and model topology as we train  
    
    # trainer.test(model, ckpt_path="best") # ckpt_path="last" to load and evaluate the last model
    from pathlib import Path
    artifact = run.use_artifact('frkangul/foreground-car-segm/model-m1l47gow:v67', type='model')
    artifact_dir = artifact.download()
    trainer.test(model, ckpt_path=str(Path(artifact_dir, "model.ckpt"))) # ckpt_path="last" to load and evaluate the last model
    
    
if __name__ == "__main__":
    test_pipeline()