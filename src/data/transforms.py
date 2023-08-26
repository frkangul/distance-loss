import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(cfg):
    """
    This function generates a dictionary of transformations for training, validation, testing, and unnormalization.
    The transformations include resizing the image to the longest side, padding if needed, normalizing with ImageNet mean and std, and converting the image to a PyTorch tensor.
    For unnormalization, the mean is negated and divided by std, and std is divided by itself to get 1.

    Args:
        cfg (DictConfig): A configuration object containing the parameters for the transformations.

    Returns:
        transforms (dict): containing the transformations for training, validation, testing, and unnormalization.
    """

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

    return transforms