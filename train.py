"""
This is the main script for training the image segmentation model. It includes the following steps:
1. Import necessary libraries
2. Set up Weights & Biases for experiment tracking
3. Define a function to set the seed for reproducibility
4. Define a function to set up the Weights & Biases logger
5. Define a function to set up PyTorch Lightning callbacks
6. Define the main pipeline function that sets up the configuration, seeds, logger, model, callbacks, and trainer.
7. Start the training process and finally test the model.

Auther: Furkan Gul
Date: 24.08.2023
"""
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection, VisionDataset
from torchvision.utils import draw_segmentation_masks
# from torchmetrics import Dice # https://github.com/Lightning-AI/metrics/tree/master/src/torchmetrics

import os, random
from typing import Callable, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
from pprint import pprint
from pycocotools.coco import COCO
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback

import albumentations as A
from albumentations.pytorch import ToTensorV2

import wandb
import warnings
from omegaconf import DictConfig, OmegaConf
import hydra

import distanceLoss
from plCallbacks import LogSegPredictionCallback
from plModel import ImageSegModel
import customDatasets
warnings.filterwarnings("ignore") # category=DeprecationWarning

WANDB_KEY="615a4a8c6b3ade78e75eba4a9c1ed70e4f564178"

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
        project=cfg.exp.wandb.proj_name,
        name=f"{cfg.exp.model_name}/{cfg.exp.encoder_name}/{cfg.dataset.data_name}/{cfg.exp.loss}/{cfg.exp.wandb.exp_name}",
        group= f"{cfg.exp.model_name}/{cfg.exp.encoder_name}/{cfg.dataset.data_name}",
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
        monitor=cfg.trainer.model_ckpt_motior,
        mode="max", # log model only if `val_per_image_iou` increases
        save_top_k=1, # to save the best model. save_top_k=-1 to save all models
        # every_n_epochs=5, # to save at every n epochs
        save_last=True,
        # To save locally:
        dirpath=cfg.trainer.ckpt_save_dir,
        filename='{epoch}-'+f'{cfg.exp.model_name}-{cfg.exp.encoder_name}-lr{cfg.trainer.lr}-hight{cfg.dataset.transform.image_resize_h}-width{cfg.dataset.transform.image_resize_w}-{cfg.dataset.data_name}'
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
    
@hydra.main(version_base=None, config_path="./conf", config_name="config")
def pipeline(cfg: DictConfig):
    """
    The main pipeline function that sets up the configuration, seeds, logger, model, callbacks, and trainer.
    It then starts the training process and finally tests the model.
    """
    pprint(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.exp.SEED)
    wandb_logger = setup_wandb_and_logger(cfg)
    
    transforms = {
        "train": A.Compose([
            # A.Resize(height=cfg.transform.image_resize_h, width=cfg.transform.image_resize_w),
            A.LongestMaxSize(max(cfg.dataset.transform.image_resize_h, cfg.dataset.transform.image_resize_w)),
            A.PadIfNeeded(min_height=cfg.dataset.transform.image_resize_h, min_width=cfg.dataset.transform.image_resize_w, border_mode=cv2.BORDER_REFLECT_101),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
            ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
            ]),
        "val": A.Compose([
            # A.Resize(height=cfg.transform.image_resize_h, width=cfg.transform.image_resize_w),
            A.LongestMaxSize(max(cfg.dataset.transform.image_resize_h, cfg.dataset.transform.image_resize_w)),
            A.PadIfNeeded(min_height=cfg.dataset.transform.image_resize_h, min_width=cfg.dataset.transform.image_resize_w, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
            ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
        ]),
        "test": A.Compose([
            A.LongestMaxSize(max(cfg.dataset.transform.image_resize_h, cfg.dataset.transform.image_resize_w)),
            A.PadIfNeeded(min_height=cfg.dataset.transform.image_resize_h, min_width=cfg.dataset.transform.image_resize_w, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
            ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
        ]),
        "unnorm": A.Compose([
            A.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1.0/0.229, 1.0/0.224, 1.0/0.225), max_pixel_value=1.0),
            ToTensorV2()
            ]) # -mean / std, 1.0 / std for unnormalization
    }

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
        logger=[CSVLogger(save_dir="logs/"), wandb_logger], # multiple loggers
        strategy=None if cfg.server.device_num==1 else 'dp', # 'dp' strategy is used when multiple-gpus with 1 machine.
        deterministic=cfg.exp.deterministic, # for reproducibity. 'warn' to use deterministic algorithms whenever possible
    )
    
    wandb_logger.watch(model, log="all") # log and monitor gradients, parameter histogram and model topology as we train  

    trainer.fit(model) # ckpt_path=cfg.trainer.ckpt_path4resume, ckpt_path="best"
    
    trainer.test(model, ckpt_path="best") # ckpt_path="last" to load and evaluate the last model


if __name__ == "__main__":
    pipeline()