import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as visF

import os
import numpy as np
import matplotlib.pyplot as plt 
from pytorch_lightning.callbacks import Callback

import wandb
import warnings
warnings.filterwarnings("ignore") # category=DeprecationWarning

# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-validation-epoch-end
class ReduceLROnPlateauOptCallback(Callback): 
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""
        # Manual optimization. Note that step should be called after val_loss is calculated
        sch = pl_module.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, ReduceLROnPlateau):
            new_lr = sch.step(pl_module.trainer.callback_metrics["val_loss"])
            if new_lr is None:
                print("Regular ReduceLROnPlateau")
            elif isinstance(new_lr, float):
                print(f"Optimized ReduceLROnPlateau with new lr of {new_lr}")
                trainer.fit(pl_module, ckpt_path="best")
                
                
        
# https://docs.wandb.ai/guides/integrations/lightning
# to control when you log to Weights & Biases via the WandbLogger
class LogSegPredictionCallback(Callback): 
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        # outputs: The outputs of validation_step_end(validation_step(x))
        # Let's log all sample image predictions from the fifth batch
        if batch_idx == pl_module.cfg.vis.val_batch_id: 
            with torch.no_grad():   
                # x, y = batch["image"], batch["mask"]
                x, y, y_distance, y_distance_sum = batch["image"], batch["mask"], batch["distance_mask"], batch["distance_mask_sum"]
                
                # All five images and preds in one figure
                fig1, axs1 = plt.subplots(pl_module.cfg.vis.img_num, 5, figsize=(50, 50)) # pl_module.cfg.vis.val_dl.batch_size
                for img_id, (img_gt, mask_gt, mask_dist_gt, mask_pred) in enumerate(zip(x, y, y_distance, outputs["prob_y"])): # [1, num_channels, height, width]
                    mask_pred_th = (mask_pred > 0.5).float()
                    un_img = img_gt.cpu().numpy().squeeze().transpose(1, 2, 0) # for visualization we have to transpose back to HWC
                    img_gt = (pl_module.unnorm_transform(image=un_img)["image"]*255).to(torch.uint8)
                    
                    axs1[img_id, 0].imshow(np.asarray(visF.to_pil_image(img_gt)))
                    axs1[img_id, 0].set_title("Image", fontsize=30)

                    axs1[img_id, 1].imshow(mask_dist_gt.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 1].set_title("GT-DistanceMask", fontsize=30)
                    
                    axs1[img_id, 2].imshow(mask_gt.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 2].set_title("GT-Mask", fontsize=30)
    
                    axs1[img_id, 3].imshow(mask_pred.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 3].set_title("Pred-Mask", fontsize=30)
                        
                    axs1[img_id, 4].imshow(mask_pred_th.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 4].set_title("Pred-Mask Thresholded", fontsize=30)
                    
                    # Save only first 4 images in the batch
                    if img_id == pl_module.cfg.vis.img_num - 1:
                        break
                plt.show()
                img_dir = os.path.join(pl_module.cfg.vis.dir, f"epoch{pl_module.current_epoch}_data_pred_vis.png")
                fig1.savefig(img_dir, bbox_inches='tight') 
                # wandb_logger.log_image(key=f"val_batch{batch_idx}_all", images=[img_dir])
                trainer.logger[1].experiment.log({f"val_batch{batch_idx}_all":[wandb.Image(img_dir, caption=f"val_batch{batch_idx}_all")]})
                
                # Four images and their preds are in different figures
                fig2, axs2 = plt.subplots(2, 2, figsize=(30, 30)) 
                for img_id, (img_gt, mask_gt, mask_pred) in enumerate(zip(x, y, outputs["prob_y"])): # [1, num_channels, height, width]
                    mask_pred_th = (mask_pred > 0.5).squeeze(0)
                    mask_gt_th = mask_gt > 0.5 # boolean
                    
                    un_img = img_gt.cpu().numpy().squeeze().transpose(1, 2, 0) # for visualization we have to transpose back to HWC
                    img = (pl_module.unnorm_transform(image=un_img)["image"]*255).to(torch.uint8)
                         
                    gt_mask_on_img = draw_segmentation_masks(image=img, masks=mask_gt_th, alpha=0.7, colors=["orange"]) # Tensor[C, H, W]
                    pred_mask_on_img = draw_segmentation_masks(image=img, masks=mask_pred_th, alpha=0.7, colors=["orange"]) # Tensor[C, H, W]
                    
                    axs2[0, 0].imshow(np.asarray(visF.to_pil_image(gt_mask_on_img)))
                    axs2[0, 0].set_title("GT-Mask on Image", fontsize=30)
                    
                    axs2[0, 1].imshow(np.asarray(visF.to_pil_image(pred_mask_on_img)))
                    axs2[0, 1].set_title("Pred-Mask on Image", fontsize=30)   
        
                    axs2[1, 0].imshow(mask_gt.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs2[1, 0].set_title("GT-Mask", fontsize=30)
                        
                    axs2[1, 1].imshow(mask_pred.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs2[1, 1].set_title("Pred-Mask Without Threshold", fontsize=30)
                    
                    plt.show()
                    
                    img_dir = os.path.join(pl_module.cfg.vis.dir, f"epoch{pl_module.current_epoch}_val_batch{batch_idx}_image{img_id}.png")
                    fig2.savefig(img_dir, bbox_inches='tight') 
                    # wandb_logger.log_image(key=f"val_batch{batch_idx}_image{img_id}", images=[img_dir])
                    trainer.logger[1].experiment.log({f"val_batch{batch_idx}_image{img_id}":[wandb.Image(img_dir, caption=f"val_batch{batch_idx}_image{img_id}")]})
                    # Save only first 4 images in the batch
                    if img_id == pl_module.cfg.vis.img_num - 1:
                        break
                        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the train batch ends."""
        # outputs: The outputs of train_step_end(train_step(x))??
        # Let's log all sample image predictions from the fifth batch
        if batch_idx == pl_module.cfg.vis.val_batch_id: 
            with torch.no_grad():   
                # x, y = batch["image"], batch["mask"]
                x, y, y_distance, y_distance_sum = batch["image"], batch["mask"], batch["distance_mask"], batch["distance_mask_sum"]
                
                # All five images and preds in one figure
                fig1, axs1 = plt.subplots(pl_module.cfg.vis.img_num, 5, figsize=(50, 50)) # pl_module.cfg.vis.val_dl.batch_size
                for img_id, (img_gt, mask_gt, mask_dist_gt, mask_pred) in enumerate(zip(x, y, y_distance, outputs["prob_y"])): # [1, num_channels, height, width]
                    mask_pred_th = (mask_pred > 0.5).float()
                    un_img = img_gt.cpu().numpy().squeeze().transpose(1, 2, 0) # for visualization we have to transpose back to HWC
                    img_gt = (pl_module.unnorm_transform(image=un_img)["image"]*255).to(torch.uint8)
                    
                    axs1[img_id, 0].imshow(np.asarray(visF.to_pil_image(img_gt)))
                    axs1[img_id, 0].set_title("Image", fontsize=30)

                    axs1[img_id, 1].imshow(mask_dist_gt.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 1].set_title("GT-DistanceMask", fontsize=30)
                    
                    axs1[img_id, 2].imshow(mask_gt.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 2].set_title("GT-Mask", fontsize=30)
    
                    axs1[img_id, 3].imshow(mask_pred.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 3].set_title("Pred-Mask", fontsize=30)
                        
                    axs1[img_id, 4].imshow(mask_pred_th.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs1[img_id, 4].set_title("Pred-Mask Thresholded", fontsize=30)
                    
                    # Save only first 4 images in the batch
                    if img_id == pl_module.cfg.vis.img_num - 1:
                        break
                plt.show()
                img_dir = os.path.join(pl_module.cfg.vis.dir, f"train_epoch{pl_module.current_epoch}_data_pred_vis.png")
                fig1.savefig(img_dir, bbox_inches='tight') 
                # wandb_logger.log_image(key=f"train_batch{batch_idx}_all", images=[img_dir])
                trainer.logger[1].experiment.log({f"train_batch{batch_idx}_all":[wandb.Image(img_dir, caption=f"train_batch{batch_idx}_all")]})
                
                # Four images and their preds are in different figures
                fig2, axs2 = plt.subplots(2, 2, figsize=(30, 30)) 
                for img_id, (img_gt, mask_gt, mask_pred) in enumerate(zip(x, y, outputs["prob_y"])): # [1, num_channels, height, width]
                    mask_pred_th = (mask_pred > 0.5).squeeze(0)
                    mask_gt_th = mask_gt > 0.5 # boolean
                    
                    un_img = img_gt.cpu().numpy().squeeze().transpose(1, 2, 0) # for visualization we have to transpose back to HWC
                    img = (pl_module.unnorm_transform(image=un_img)["image"]*255).to(torch.uint8)
                         
                    gt_mask_on_img = draw_segmentation_masks(image=img, masks=mask_gt_th, alpha=0.7, colors=["orange"]) # Tensor[C, H, W]
                    pred_mask_on_img = draw_segmentation_masks(image=img, masks=mask_pred_th, alpha=0.7, colors=["orange"]) # Tensor[C, H, W]
                    
                    axs2[0, 0].imshow(np.asarray(visF.to_pil_image(gt_mask_on_img)))
                    axs2[0, 0].set_title("GT-Mask on Image", fontsize=30)
                    
                    axs2[0, 1].imshow(np.asarray(visF.to_pil_image(pred_mask_on_img)))
                    axs2[0, 1].set_title("Pred-Mask on Image", fontsize=30)   
        
                    axs2[1, 0].imshow(mask_gt.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs2[1, 0].set_title("GT-Mask", fontsize=30)
                        
                    axs2[1, 1].imshow(mask_pred.cpu().numpy().squeeze())  # for visualization we have to remove 3rd dimension of mask
                    axs2[1, 1].set_title("Pred-Mask Without Threshold", fontsize=30)
                    
                    plt.show()
                    
                    img_dir = os.path.join(pl_module.cfg.vis.dir, f"epoch{pl_module.current_epoch}_train_batch{batch_idx}_image{img_id}.png")
                    fig2.savefig(img_dir, bbox_inches='tight') 
                    # wandb_logger.log_image(key=f"train_batch{batch_idx}_image{img_id}", images=[img_dir])
                    trainer.logger[1].experiment.log({f"train_batch{batch_idx}_image{img_id}":[wandb.Image(img_dir, caption=f"train_batch{batch_idx}_image{img_id}")]})
                    
                    # Save only first 4 images in the batch
                    if img_id == pl_module.cfg.vis.img_num - 1:
                        break
                    
                    
                    
                    
# WANDB SEGMENTASYON GÖRSELLEŞTİRMELERİ PEK İYİ DEĞİL
# https://docs.wandb.ai/guides/track/log/media#image-overlays
# to control when you log to Weights & Biases via the WandbLogger
class LogSegPredOverlayCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
        # outputs: The outputs of validation_step_end(validation_step(x))
        # Let's log all sample image predictions from the fifth batch
        if batch_idx == pl_module.cfg.vis.val_batch_id:
            # self.eval()
            with torch.no_grad():   
                x, y = batch["image"], batch["mask"]
                images = [img for img in x]
                # for visualization in wandb, masks must be in numpy format without 3rd dimension
                masks = [{"predictions": {"mask_data": (mask_pred > 0.5).float().cpu().detach().numpy().squeeze(), "class_labels": {1: "car"}},
                          "ground_truth": {"mask_data": mask_gt.cpu().detach().numpy().squeeze(), "class_labels": {1: "car"}}}
                          for mask_gt, mask_pred in zip(y, outputs["prob_y"])]
            wandb_logger.log_image(key=f"val_batch{batch_idx}_image{img_id}", images=images, masks=masks)
            # self.logger.experiment.log({f"val_batch{batch_idx}_image{img_id}":[wandb.Image(img_dir, caption=f"val_batch{batch_idx}_image{img_id}")]})
            
        

class ReduceLROnPlateauOptimized(ReduceLROnPlateau):
    """
    Bunu implemente edince ReduceLROnPlateau ile aynı şekilde çalışıyor
    """
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr)) 
                return new_lr
            
    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            new_lr = self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        if 'new_lr' in locals():
            return new_lr