"""
This module contains the implementation of DistanceLoss, a loss function for image segmentation tasks. 
It supports binary, multiclass and multilabel cases. The module also includes utility functions 
for converting masks to boundary tensors and calculating soft distance scores.

Classes:
    DistanceLoss: A class that calculates the distance loss for image segmentation tasks.

Functions:
    to_tensor(x, dtype=None): Converts the input to a torch.Tensor.
    soft_distance_score(output, target, target_sum, smooth=0.0, eps=1e-7, dims=None): Calculates the soft distance score.
    mask_to_boundary_tensor(mask, dilation_ratio=0.02): Converts a binary mask to a boundary mask.
"""

from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
import cv2

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/constants.py

#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"

# https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    
def soft_distance_score(
    output: torch.Tensor,
    target: torch.Tensor,
    target_sum: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
    else:
        intersection = torch.sum(output * target)   
    distance_score = intersection / (target_sum + smooth).clamp_min(eps)
    return distance_score

__all__ = ["DistanceLoss"]

class DistanceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Distance loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            # https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DistanceLoss, self).__init__()

        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_true_sum: torch.Tensor, y_org_mask: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (2) # make operations over each image and each class seperately

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)
            y_true_sum = y_true_sum.view(bs, 1) # y_true_sum.unsqueeze(dim=-1) # from the size of (bs) to the size of (bs, 1) for shape compatibility
            y_org_mask = y_org_mask.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)
            y_true_sum = y_true_sum.view(bs, num_classes) # y_true_sum.squeeze() # from the size of (bs, num_classes, 1) to the size of (bs, num_classes) for shape compatibility
            y_org_mask = y_org_mask.view(bs, num_classes, -1)

        y_true_sum_updated = y_true_sum + torch.sum((1 - y_org_mask)*y_pred, dim=dims) # to penalize the wrong preds outside of objects

        scores = soft_distance_score(
            y_pred,
            y_true.type_as(y_pred), # y_true.type(y_pred.dtype)
            target_sum=y_true_sum_updated,
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        
        mask = y_org_mask.sum(dims) > 0 # y_true_sum.sum() > 0, y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)
            
        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()

# BOUNDARY IOU TENSOR IMPLEMENTATION
# General util function to get the boundary of a binary mask.
def mask_to_boundary_tensor(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    Arguments:
        mask (torch.Tensor): mask of torch.Tensor of shape (N, C, H, W)
        dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    Returns: 
        boundary mask (torch.Tensor): boundary mask of torch.Tensor of shape (N, C, H, W)
    """
    nb, c, h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
        
    boundary_mask = torch.zeros_like(mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for idx, img_tensor in enumerate(mask[:,]): # img_tensor will be in the shape of (C, H, W)
        if c > 1: # multiclass case
            for c_idx, img_tensor_ in enumerate(img_tensor[:,]): # img_tensor_ will be in the shape of (H, W)
                # From tensor to numpy
                img_np = np.array(img_tensor_.cpu().detach(), dtype="uint8")
                # Pad image so mask truncated by the image border is also considered as boundary.
                new_mask = cv2.copyMakeBorder(img_np, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
                mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1] # in the shape of (H, W)
                boundary_mask_np = img_np - mask_erode
                boundary_mask_np = np.expand_dims(boundary_mask_np, 0)
                # import pdb; pdb.set_trace()
                # From numpy to tensor
                boundary_mask[idx, c_idx] = torch.tensor(boundary_mask_np)
        elif c == 1: # binary case
            # From tensor to numpy
            img_np = np.array(img_tensor.cpu().detach(), dtype="uint8").transpose(1, 2, 0).squeeze() # in the shape of (H, W, C) -> (H, W) since C=1
            # Pad image so mask truncated by the image border is also considered as boundary.
            new_mask = cv2.copyMakeBorder(img_np, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
            mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1] # in the shape of (H, W)
            boundary_mask_np = img_np - mask_erode
            boundary_mask_np = np.expand_dims(boundary_mask_np, 0)
            # import pdb; pdb.set_trace()
            # From numpy to tensor
            boundary_mask[idx,] = torch.tensor(boundary_mask_np)
    return boundary_mask