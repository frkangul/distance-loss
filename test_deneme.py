from src.data.customDatasets import DatasetFromSubset, CocoToSmpDataset, CityscapesToSmpDataset, random_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms():
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
            A.LongestMaxSize(max(512, 512)),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_REFLECT_101),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
            ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
            ]),
        "val": A.Compose([
            # A.Resize(height=cfg.transform.image_resize_h, width=cfg.transform.image_resize_w),
            A.LongestMaxSize(max(512, 512)),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
            ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
        ]),
        "test": A.Compose([
            A.LongestMaxSize(max(512, 512)),
            A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # use ImageNet image normalization 
            ToTensorV2() # numpy HWC image is converted to pytorch CHW tensor
        ]),
        "unnorm": A.Compose([
            A.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), std=(1.0/0.229, 1.0/0.224, 1.0/0.225), max_pixel_value=1.0),
            ToTensorV2()
            ]) # -mean / std, 1.0 / std for unnormalization
    }

    return transforms

transforms = get_transforms()
test_transform = transforms["test"]

test_ds = CityscapesToSmpDataset("/home/frkangul/distance-loss/data/cityscapes", split='test', mode='fine', transforms=test_transform)

print(len(test_ds))