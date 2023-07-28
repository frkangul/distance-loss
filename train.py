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
from box import Box

import distanceLoss
from plCallbacks import LogSegPredictionCallback
from plModel import ImageSegModel
import customDatasets
warnings.filterwarnings("ignore") # category=DeprecationWarning


cfg = {
    "max_epoch": 60,
    "max_time": "02:23:55:00", 
    "distance_transform_loss": False,
    "wandb": {
        "exp_name": "bce",
        "proj_name": "foreground-car-segm",
    },
    "model_ckpt_motior": "val_per_image_bIoU",
    "data_name": "CelebAMask-HQ",
    "output_class_num": 1,
    "mode": "binary",
    "data_dir": "./data/CelebAMask-HQ-v2/",
    "trainer": {
        "accelerator": "auto", # gpu, tpu, cpu, auto
        "device_num": 1, # Number of gpu, check if T4 vs P100
        "precision": 16, # Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16)
        "ckpt_path4resume": "model.ckpt",
    },
    "transform": {
        "image_resize_h": 512, # 576
        "image_resize_w": 512, # 800
    },
    "model": {
        "model_name": "Unet", # "DeepLabV3Plus", "UnetPlusPlus", "Unet"
        "encoder_name": "resnet50", # "efficientnet-b5","resnet34", "mit_b1"       
    },
    "optimizer": {
        "lr": 0.0003, # 0.001, 0.0003
        "reduce_rl_on": False,
    },   
    "patience": 20, # for validation loss
    "train_dl": {
        "batch_size": 128,
    },
    "val_dl": {
        "batch_size": 128,
    },
    "test_dl": {
        "batch_size": 128,
    },
    "SEED": 42,
    "vis_img_num": 8,
    "vis_val_batch_id": 5,
    "vis_dir": "./resulting_imgs/CelebAMask-HQ_512_bce_unet_r50", 
    "ckpt_save_dir": './logs/lightning_logs/checkpoints/'
}

WANDB_KEY="615a4a8c6b3ade78e75eba4a9c1ed70e4f564178"


def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    cfg = Box(cfg)
    pprint(cfg)
    
    wandb.login(key=WANDB_KEY)
    seed_everything(cfg.SEED)
    
    train_transform = A.Compose([
        # A.Resize(height=cfg.transform.image_resize_h, width=cfg.transform.image_resize_w),
        A.LongestMaxSize(max(cfg.transform.image_resize_h, cfg.transform.image_resize_w)),
        A.PadIfNeeded(min_height=cfg.transform.image_resize_h, min_width=cfg.transform.image_resize_w, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
        ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
        ])

    val_transform = A.Compose([
        # A.Resize(height=cfg.transform.image_resize_h, width=cfg.transform.image_resize_w),
        A.LongestMaxSize(max(cfg.transform.image_resize_h, cfg.transform.image_resize_w)),
        A.PadIfNeeded(min_height=cfg.transform.image_resize_h, min_width=cfg.transform.image_resize_w, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
        ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
    ])

    test_transform = A.Compose([
        A.LongestMaxSize(max(cfg.transform.image_resize_h, cfg.transform.image_resize_w)),
        A.PadIfNeeded(min_height=cfg.transform.image_resize_h, min_width=cfg.transform.image_resize_w, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
        ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
    ])

    unnorm_transform = A.Compose([
        A.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1.0/0.229, 1.0/0.224, 1.0/0.225), max_pixel_value=1.0),
        ToTensorV2()
        ]) # -mean / std, 1.0 / std for unnormalization
    
    model = ImageSegModel(cfg, train_transform, val_transform, test_transform, unnorm_transform)
    ModelSummary(model) #to see detailed layer based parameter nums max_depth=-1
    
    wandb_logger = WandbLogger(
        project=cfg.wandb.proj_name,
        name=f"{cfg.model.model_name}/{cfg.model.encoder_name}/{cfg.wandb.exp_name}",
        group= f"{cfg.model.model_name}/{cfg.model.encoder_name}/{cfg.data_name}",
        log_model="all", # model checkpoints are logged during training
    )
    wandb_logger.watch(model, log="all") # log and monitor gradients, parameter histogram and model topology as we train  
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

    model_checkpointer = ModelCheckpoint(
        monitor=cfg.model_ckpt_motior,
        mode="max", # log model only if `val_per_image_iou` increases
        save_top_k=1, # to save the best model. save_top_k=-1 to save all models
        # every_n_epochs=5, # to save at every n epochs
        save_last=True,
        # To save locally:
        dirpath=cfg.ckpt_save_dir,
        filename='{epoch}-'+f'{cfg.model.model_name}-{cfg.model.encoder_name}-lr{cfg.optimizer.lr}-hight{cfg.transform.image_resize_h}-width{cfg.transform.image_resize_w}-{cfg.data_name}'
    )

    earlystop_checkpointer = EarlyStopping(
        monitor="val_loss", mode="min", patience=cfg.patience, verbose=True
    ) # verbose = 0, means silent.

    lr_monitor = LearningRateMonitor() # logging_interval='epoch'/'step'. Set to None to log at individual interval according to the interval key of each scheduler
    
    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch,
        max_time=cfg.max_time, # Stop after max_time hours of training or when reaching max_epochs epochs
        precision=cfg.trainer.precision, # Mixed Precision (16-bit) combines the use of both 32 and 16-bit floating points to reduce memory footprint
        accelerator=cfg.trainer.accelerator, # gpu, tpu, auto
        devices=cfg.trainer.device_num,
        callbacks=[TQDMProgressBar(refresh_rate=20), # for notebook usage
                # earlystop_checkpointer,
                model_checkpointer,
                LogSegPredictionCallback(),
                lr_monitor,
                # ReduceLROnPlateauOptCallback()
                ],
        logger=[CSVLogger(save_dir="logs/"), wandb_logger], # multiple loggers
        strategy=None if cfg.trainer.device_num==1 else 'dp', # 'dp' strategy is used when multiple-gpus with 1 machine.
        deterministic=True, # for reproducibity
    )
    
    trainer.fit(model) # ckpt_path=cfg.trainer.ckpt_path4resume, ckpt_path="best"
    
    trainer.test(model, ckpt_path="best") # ckpt_path="last" to load and evaluate the last model
    
    wandb.finish()