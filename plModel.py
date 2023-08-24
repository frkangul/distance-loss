import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection, VisionDataset
from torchvision.utils import draw_segmentation_masks
# from torchmetrics import Dice # https://github.com/Lightning-AI/metrics/tree/master/src/torchmetrics

from typing import Callable, Optional, Tuple
from pathlib import Path
from pprint import pprint
from pycocotools.coco import COCO

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback

import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import os
from distanceLoss import DistanceLoss, mask_to_boundary_tensor
from customDatasets import DatasetFromSubset, CocoToSmpDataset, random_split

BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"

# https://pytorch-lightning.readthedocs.io/en/stable/guides/speed.html
class ImageSegModel(pl.LightningModule):
    """
    This is just regular PyTorch organized in a specific format.

    Notice the following:

    * no GPU/TPU specific code
    * no .to(device)
    * np .cuda()
    """
    def __init__(self, cfg, train_transform, val_transform, test_transform, unnorm_transform):
        '''method used to define our model and data parameters'''
        super().__init__()    
        # Set our init args as class attributes
        self.n_cpu = os.cpu_count()
        self.cfg = cfg
        
        # Define PyTorch model, model weights are in cuda
        encoder_weights = "imagenet"
        input_channels = 3
        self.model = smp.create_model(arch=cfg.model.model_name,
                                      encoder_name=cfg.model.encoder_name,
                                      encoder_weights=encoder_weights,
                                      in_channels=input_channels,
                                      classes=cfg.output_class_num)
        
        # Preprocessing/Transformations  
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.unnorm_transform = unnorm_transform

        # Define loss. If predicted mask contains logits, and loss param `from_logits` is set to True
        if cfg.loss == "dist_transform":
            self.loss = DistanceLoss(BINARY_MODE, from_logits=True)
        elif cfg.loss == "bce":
            self.loss = BCEWithLogitsLoss()
        elif cfg.loss == "iou":
            self.loss = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif cfg.loss == "dice":
            self.loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif cfg.loss == "dice&bce":
            self.loss_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
            self.loss_bce = BCEWithLogitsLoss()
        elif cfg.loss == "dice&focal":
            self.loss_dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
            self.loss_focal = smp.losses.FocalLoss(smp.losses.BINARY_MODE) # it uses focal_loss_with_logits
        self.distance_transform_metric = DistanceLoss(BINARY_MODE, from_logits=False)
        
        self.save_hyperparameters(ignore=["model"])
    
    def forward(self, x):
        '''method used for inference input -> output'''
        return self.model(x) # logits

    def _shared_step(self, batch, stage):
        '''needs to return a loss from a single batch'''
        x, y, y_distance, y_distance_sum = batch["image"], batch["mask"], batch["distance_mask"], batch["distance_mask_sum"]
            
        # Shape of the image should be [batch_size, num_channels, height, width]
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert x.ndim == 4
    
        # Shape of the mask should be [batch_size, num_classes, height, width]. For binary segmentation num_classes=1
        assert y.ndim == 4
        assert y_distance.ndim == 4
        
        # Check that image dimensions are divisible by 32, encoder and decoder connected by `skip connections`.
        # Usually encoder have 5 stages of downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80 and we will get an error trying to concat these features
        h, w = x.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert y.max() <= 1.0 and y.min() >= 0
        assert y_distance.max() <= 1.0 and y_distance.min() >= -1    
        
        logits_y = self(x) # it is actually self.forward(x)
        # Lets compute metrics for some threshold. First convert mask values to probabilities, then apply thresholding
        prob_y = logits_y.sigmoid()
        pred_y_th = (prob_y > 0.95).float()
        
        boundary_y = mask_to_boundary_tensor(y, dilation_ratio=0.02)
        boundary_pred_y = mask_to_boundary_tensor(prob_y, dilation_ratio=0.02)
   
        if self.cfg.loss == "dist_transform":
            loss = self.loss(logits_y, y_distance, y_distance_sum)
        elif self.cfg.loss == "dice&bce":
            loss = self.loss_dice(logits_y, y) + self.loss_bce(logits_y, y)
        elif self.cfg.loss == "dice&focal":
            loss = self.loss_dice(logits_y, y) + self.focal(logits_y, y)
        else: # when using single loss function
            loss = self.loss(logits_y, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_y_th.long(), y.long(), mode=self.cfg.mode, num_classes=self.cfg.output_class_num)
        
        # Extract tp, fp, fn, tn for boundary iou 
        boundary_pred_y_th = mask_to_boundary_tensor(pred_y_th, dilation_ratio=0.02)
        b_tp, b_fp, b_fn, b_tn = smp.metrics.get_stats(boundary_pred_y_th.long(), boundary_y.long(), mode="binary")
        
        # Calculate distance transform evaluation metric here. Do not concaterate in the epoch_end and then evaluate since it will require more computation.
        distance_transform_metric = 1- self.distance_transform_metric(pred_y_th, y_distance, y_distance_sum)
        self.log(f"{stage}_distance_transform_evalmetric", distance_transform_metric, on_step=False, on_epoch=True, prog_bar=True)
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "b_tp": b_tp,
            "b_fp": b_fp,
            "b_fn": b_fn,
            "b_tn": b_tn,
            "prob_y": prob_y
        }
     
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
 
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def _shared_step_end(self, batch_parts):
        # Check if there is multi-gpu or not. This step must be here due to pl lib.
        if batch_parts["loss"].numel() != 1: # multi-gpu case
            batch_parts["loss"] = torch.mean(batch_parts["loss"], dim=0) # take the mean of losses from different GPUs
        return batch_parts
    
    def training_step_end(self, batch_parts):
        return self._shared_step_end(batch_parts) 

    def validation_step_end(self, batch_parts):
        return self._shared_step_end(batch_parts) 
    
    def test_step_end(self, batch_parts):
        return self._shared_step_end(batch_parts) 

    def _shared_epoch_end(self, step_outputs, stage):
        # outputs are coming from the result or trainin/validation/test steps respectively
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in step_outputs])
        fp = torch.cat([x["fp"] for x in step_outputs])
        fn = torch.cat([x["fn"] for x in step_outputs])
        tn = torch.cat([x["tn"] for x in step_outputs])
        b_tp = torch.cat([x["b_tp"] for x in step_outputs])
        b_fp = torch.cat([x["b_fp"] for x in step_outputs])
        b_fn = torch.cat([x["b_fn"] for x in step_outputs])
        b_tn = torch.cat([x["b_tn"] for x in step_outputs])

        # per image IoU means that we first calculate IoU score for each image and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset and then compute IoU score. 
        # The difference between dataset_iou and per_image_iou scores in this particular case will not be much, 
        # however for dataset with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        per_image_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        
        per_image_fbeta = smp.metrics.fbeta_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_fbeta = smp.metrics.fbeta_score(tp, fp, fn, tn, reduction="micro")
        
        per_image_sensitivity= smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_sensitivity = smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro")
        
        per_image_specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_specificity = smp.metrics.specificity(tp, fp, fn, tn, reduction="micro")
        
        per_image_bIoU = smp.metrics.iou_score(b_tp, b_fp, b_fn, b_tn, reduction="micro-imagewise")
        dataset_bIoU = smp.metrics.iou_score(b_tp, b_fp, b_fn, b_tn, reduction="micro")
        
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_per_image_dice": per_image_dice,
            f"{stage}_dataset_dice": dataset_dice,
            f"{stage}_per_image_fbeta": per_image_fbeta,
            f"{stage}_dataset_fbeta": dataset_fbeta,
            f"{stage}_per_image_sensitivity": per_image_sensitivity,
            f"{stage}_dataset_sensitivity": dataset_sensitivity,
            f"{stage}_per_image_specificity": per_image_specificity,
            f"{stage}_dataset_specificity": dataset_specificity,
            f"{stage}_per_image_bIoU": per_image_bIoU,
            f"{stage}_dataset_bIoU": dataset_bIoU,
        }
        
        self.log_dict(metrics, prog_bar=True)
    
    def training_epoch_end(self, training_step_outputs):
        return self._shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self._shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self._shared_epoch_end(test_step_outputs, "test")  
            
    def configure_optimizers(self):
        '''defines model optimizer and scheduler'''
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        if self.cfg.optimizer.reduce_rl_on:          
            # It is called automatic optimization if `scheduler.step()` is not called manually inside pl.LightningModule. Otherwise, it is manuel optimization
            scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-7) # ReduceLROnPlateauOptimized
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch", # The unit of the scheduler's step size, could also be 'step'. 'epoch' updates the scheduler on epoch end whereas 'step' updates it after a optimizer update.
                    "monitor": "val_loss", # Metric to monitor for schedulers like ReduceLROnPlateau
                    "frequency": 1 # How many epochs/steps should pass between calls to `scheduler.step()`. 1 corresponds to updating the learning rate after every epoch/step. 
                    # If "monitor" references validation metrics, then "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch".
                    # "frequency" and "interval" will be ignored even if they are provided in here configure_optimizers() during manual optimization
                },
            }
        else:
            return optimizer
    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            full_ds = CocoToSmpDataset(root=os.path.join(self.cfg.data_dir, "train"), 
                                       annFile=os.path.join(self.cfg.data_dir, "annotations_train.json")
                                       )
            # Train-val split before appliying transformations
            train_subset, val_subset = random_split(full_ds, [0.8, 0.2],
                                                      generator=torch.Generator().manual_seed(self.cfg.SEED))
            # Apply transformations to each subset
            self.train_ds = DatasetFromSubset(train_subset, transforms=self.train_transform)
            self.val_ds = DatasetFromSubset(val_subset, transforms=self.val_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_ds = CocoToSmpDataset(root=os.path.join(self.cfg.data_dir, "test"), 
                                            annFile=os.path.join(self.cfg.data_dir, "annotations_test.json"),
                                            transforms=self.test_transform)
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.train_dl.batch_size, shuffle=True, num_workers=self.n_cpu, pin_memory=True)
        # pin_memory will put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.val_dl.batch_size, shuffle=False, num_workers=self.n_cpu, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.cfg.test_dl.batch_size, shuffle=False, num_workers=self.n_cpu, pin_memory=True)