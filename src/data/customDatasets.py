"""
This module contains custom dataset classes and utility functions for data manipulation.

Classes:
    CocoToSmpDataset: A class to handle the MS Coco Detection dataset.
    DatasetFromSubset: A class to handle subsets of a dataset.

Functions:
    random_split: Function to randomly split a dataset into non-overlapping new datasets of given lengths. COPIED DIRECTLY FROM LATEST PYTORCH SOURCE
"""
import math
from torch.utils.data import Dataset
from torch import Generator
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union
)
import warnings
warnings.filterwarnings("ignore") # category=DeprecationWarning
from torchvision.datasets import VisionDataset
from typing import Any, Callable, List, Optional, Tuple
from scipy import ndimage
import numpy as np
from PIL import Image
import os


T = TypeVar('T')

def random_split(dataset: Dataset[T], lengths: Sequence[Union[int, float]],
                 generator: Optional[Generator] = default_generator) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


# https://pytorch.org/vision/main/_modules/torchvision/datasets/coco.html#CocoDetection
class CocoToSmpDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    @staticmethod
    def _transform_binarymask_to_distance_mask(gt_mask):
        """
        Arguments:
            gt_mask (ndarray): ground-truth binary mask in HW format
        Returns:
            distance_weight (ndarray): normalized between 0 and 1 and outer of object is 1
            distance_weight_sum (float): sum af all normalized pixel values within inside of object in binary mask
        """
        # PART 1: Object distance transform
        dist = ndimage.distance_transform_cdt(gt_mask) # distance_transform_bf is less efficient
        reverse_dist = dist.max() - dist
        reverse_dist = (reverse_dist * gt_mask) # make outer background zero
        
        # CONSIDER EDGE CASES LIKE EMPTY AND FULL MASK
        if reverse_dist.max() == 0:
            if gt_mask.max() == 0: # Grount-truth mask is empty (hiç bir pixelde etiket yok)
                # every element in dist is 0
                # IoU/Distance loss is defined for non-empty classes
                distance_weight = gt_mask # full of 0s. It doesn't matter since it will be zerout out in loss func
                distance_weight_sum = np.sum(0)
                return distance_weight, distance_weight_sum
            elif gt_mask.min() == 1 : # Grount-truth mask is full (fotoğrafın her pixeli etiket)
                # every element in dist is -1
                distance_weight = gt_mask # full of 1s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
            else:
                # If 1s in mask is really less. Ie: small objects. Eg: gt_mask.sum() = 13
                distance_weight = gt_mask # very less 1s, too many 0s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
        
        inner_reverse_dist_n = reverse_dist/ reverse_dist.max() # normalize it
        # pprint(f"Inner distance transform metrics: {inner_reverse_dist_n.min()}, {inner_reverse_dist_n.max()}, {inner_reverse_dist_n.mean()}")

        # PART 2: Outer distance transform
        reverse_gt_mask = 1- gt_mask

        # PART 3: Union of inner and outer transforms
        distance_weight_vis = inner_reverse_dist_n + reverse_gt_mask # + outer_reverse_dist_n
        # plt.imshow(distance_weight_vis)
        distance_weight = inner_reverse_dist_n + reverse_gt_mask * -1 # + outer_reverse_dist_n * -1
        distance_weight_sum = np.sum(inner_reverse_dist_n)

        return distance_weight, distance_weight_sum

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = np.array(Image.open(os.path.join(self.root, path)).convert("RGB"))
        # If you need to convert to other format HWC -> CHW: img.transpose((-1, 0, 1))
        return img
    
    def _load_mask(self, id: int) -> List[Any]:
        if not self.coco.getAnnIds(id): # check for empty masks
            # Empty mask case
            h = self.coco.loadImgs(id)[0]["height"]
            w = self.coco.loadImgs(id)[0]["width"]
            arr_2dim = np.zeros((h, w)).astype(np.float32)
            return arr_2dim # in HW format
        else:
             # Non-empty mask case
            ann = self.coco.loadAnns(self.coco.getAnnIds(id))[0]
            arr_2dim = self.coco.annToMask(ann=ann).astype(np.float32)
            return arr_2dim # in HW format

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        mask = self._load_mask(id)
        
        sample = dict(image=image, mask=mask)
        if self.transforms is not None:
            sample = self.transforms(**sample)    
            sample["distance_mask"], sample["distance_mask_sum"] = self._transform_binarymask_to_distance_mask(np.array(sample["mask"])) # change binary mask to distance transformed mask
            sample["distance_mask"] = np.expand_dims(sample["distance_mask"], 0) # convert to CHW format ie. HW -> 1HW:
            sample["mask"] = np.expand_dims(sample["mask"], 0) # convert to CHW format ie. HW -> 1HW:    
        return sample
    
    def __len__(self) -> int:
        return len(self.ids)
    
    
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transforms=None):
        self.subset = subset
        self.transforms = transforms

    @staticmethod
    def _transform_binarymask_to_distance_mask(gt_mask):
        """
        Arguments:
            gt_mask (ndarray): ground-truth binary mask in HW format
        Returns:
            distance_weight (ndarray): normalized between 0 and 1 and outer of object is 1
            distance_weight_sum (float): sum af all normalized pixel values within inside of object in binary mask
        """
        # PART 1: Object distance transform
        dist = ndimage.distance_transform_cdt(gt_mask) # distance_transform_bf is less efficient
        reverse_dist = dist.max() - dist
        reverse_dist = (reverse_dist * gt_mask) # make outer background zero
        
        # CONSIDER EDGE CASES LIKE EMPTY AND FULL MASK
        if reverse_dist.max() == 0:
            if gt_mask.max() == 0: # Grount-truth mask is empty (hiç bir pixelde etiket yok)
                # every element in dist is 0
                # IoU/Distance loss is defined for non-empty classes
                distance_weight = gt_mask # full of 0s. It doesn't matter since it will be zerout out in loss func
                distance_weight_sum = np.sum(0)
                return distance_weight, distance_weight_sum
            elif gt_mask.min() == 1 : # Grount-truth mask is full (fotoğrafın her pixeli etiket)
                # every element in dist is -1
                distance_weight = gt_mask # full of 1s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
            else:
                # If 1s in mask is really less. Ie: small objects. Eg: gt_mask.sum() = 13
                distance_weight = gt_mask # very less 1s, too many 0s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
        
        inner_reverse_dist_n = reverse_dist/ reverse_dist.max() # normalize it
        # pprint(f"Inner distance transform metrics: {inner_reverse_dist_n.min()}, {inner_reverse_dist_n.max()}, {inner_reverse_dist_n.mean()}")

        # PART 2: Outer distance transform
        reverse_gt_mask = 1- gt_mask

        # PART 3: Union of inner and outer transforms
        distance_weight_vis = inner_reverse_dist_n + reverse_gt_mask # + outer_reverse_dist_n
        # plt.imshow(distance_weight_vis)
        distance_weight = inner_reverse_dist_n + reverse_gt_mask * -1 # + outer_reverse_dist_n * -1
        distance_weight_sum = np.sum(inner_reverse_dist_n)

        return distance_weight, distance_weight_sum
    
    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transforms:
            sample = self.transforms(**sample)  

        sample["distance_mask"], sample["distance_mask_sum"] = self._transform_binarymask_to_distance_mask(np.array(sample["mask"])) # change binary mask to distance transformed mask
        sample["distance_mask"] = np.expand_dims(sample["distance_mask"], 0) # convert to CHW format ie. HW -> 1HW:        
        sample["mask"] = np.expand_dims(sample["mask"], 0) # convert to CHW format ie. HW -> 1HW:
        return sample

    def __len__(self):
        return len(self.subset)