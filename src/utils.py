"""
This module contains utility functions for setting up the training environment, including seeding for reproducibility, 
logging setup with Weights & Biases, and setting up PyTorch Lightning callbacks for the training process.

Functions:
    seed_everything(seed: int): Sets the seed for generating random numbers.
    setup_wandb_and_logger(cfg): Logs in to Weights & Biases, sets up the logger, and defines the metrics to track.
    setup_pl_callbacks(cfg): Sets up the PyTorch Lightning callbacks for the training process.
"""

import os
import random
import numpy as np
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from src.plCallbacks import LogSegPredictionCallback

WANDB_KEY= os.environ["WANDB"]

def seed_everything(seed: int):
    """
    Sets the seed for generating random numbers. This is used for reproducibility in experiments.

    Args:
        seed (int): The seed value to be set for generating random numbers.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clear_wandb_artifacts(wandb_proj, artifact):
    """
    Clears unnecessary model files from the Weights & Biases artifacts.

    Args:
        wandb_proj (str): The Weights & Biases project name.
        artifact (str): The name of the artifact to delete the model files from.
    """
    
    wandb.login(key=WANDB_KEY)

    # Delete unnecessary model files from WANDB artifacts
    api = wandb.Api(overrides={
            "project": wandb_proj
            })

    artifact_type, artifact_name = "model", artifact 
    for version in api.artifact_versions(artifact_type, artifact_name):
        print(version.aliases)
        if len(version.aliases) == 0:
            version.delete()

def setup_wandb_and_logger(cfg):
    """
    Logs in to Weights & Biases using the provided key, sets up the logger, and defines the metrics to track.

    Returns:
        wandb_logger (WandbLogger): The Weights & Biases logger.
    """
    wandb.login(key=WANDB_KEY)

    # WandbLogger automatically handles the start and end of the Weights & Biases run. 
    # It calls wandb.init() when the Trainer starts and wandb.finish() when the Trainer finishes
    wandb_logger = WandbLogger(
        project=cfg.exp.wandb_proj,
        name=f"{cfg.exp.model}/{cfg.exp.encoder}/{cfg.dataset.name}/{cfg.exp.loss}/{cfg.exp.name}",
        group= f"{cfg.exp.model}/{cfg.exp.encoder}/{cfg.dataset.name}",
        log_model="all", # model checkpoints are logged during training
    )

    # W&B summary metric to display the min, max, mean or best value for that metric
    wandb.define_metric('train_per_image_dice', summary='max')
    wandb.define_metric('val_per_image_dice', summary='max')
    wandb.define_metric('train_per_image_iou', summary='max')
    wandb.define_metric('val_per_image_iou', summary='max')
    wandb.define_metric('train_per_image_bIoU', summary='max')
    wandb.define_metric('val_per_image_bIoU', summary='max')
    wandb.define_metric('train_distance_transform_evalmetric', summary='max')
    wandb.define_metric('val_distance_transform_evalmetric', summary='max')
    wandb.define_metric('train_loss', summary='min')
    wandb.define_metric('val_loss', summary='min')
    return wandb_logger

def setup_pl_callbacks(cfg):
    """
    Sets up the PyTorch Lightning callbacks for the training process. These include model checkpointing, 
    early stopping, learning rate monitoring, and others.

    Returns:
        callbacks (list): A list of PyTorch Lightning callbacks.
    """ 
    model_checkpointer = ModelCheckpoint(
        monitor=cfg.checkpoint.motior,
        mode="max", # log model only if `val_per_image_iou` increases
        save_top_k=1, # to save the best model. save_top_k=-1 to save all models
        # every_n_epochs=5, # to save at every n epochs
        save_last=True,
        # To save locally:
        dirpath=cfg.checkpoint.save_dir,
        filename='{epoch}-'+f'{cfg.exp.model}-{cfg.exp.encoder}-lr{cfg.trainer.lr}-hight{cfg.dataset.transform.image_resize_h}-width{cfg.dataset.transform.image_resize_w}-{cfg.dataset.name}'
    )

    earlystop_checkpointer = EarlyStopping(
        monitor="val_loss", mode="min", patience=cfg.trainer.patience, verbose=True
    ) # verbose = 0, means silent.

    lr_monitor = LearningRateMonitor() # logging_interval='epoch'/'step'. Set to None to log at individual interval according to the interval key of each scheduler
    
    callbacks = [
        TQDMProgressBar(refresh_rate=20), # for notebook usage
        # earlystop_checkpointer,
        model_checkpointer,
        LogSegPredictionCallback(),
        lr_monitor,
        # ReduceLROnPlateauOptCallback()
    ]
    return callbacks

if __name__ == "__main__":
    clear_wandb_artifacts("foreground-car-segm", "model-svwttkb9")