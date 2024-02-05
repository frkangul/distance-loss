"""
This module contains custom dataset classes and utility functions for data manipulation.

Classes:
    CocoToSmpDataset: A class to handle the MS Coco Detection dataset.
    DatasetFromSubset: A class to handle subsets of a dataset.

Functions:
    random_split: Function to randomly split a dataset into non-overlapping new datasets of given lengths. COPIED DIRECTLY FROM LATEST PYTORCH SOURCE
"""
import math
import torch
from torch.utils.data import Dataset
from torch import Generator
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
from typing import List, Optional, Sequence, Tuple, TypeVar, Union
import warnings

warnings.filterwarnings("ignore")  # category=DeprecationWarning
from torchvision.datasets import VisionDataset, Cityscapes
from typing import Any, Callable, List, Optional, Tuple
from scipy import ndimage
import numpy as np
from PIL import Image
import os


T = TypeVar("T")


def random_split(
    dataset: Dataset[T],
    lengths: Sequence[Union[int, float]],
    generator: Optional[Generator] = default_generator,
) -> List[Subset[T]]:
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
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(_accumulate(lengths), lengths)
    ]


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
        dist = ndimage.distance_transform_cdt(
            gt_mask
        )  # distance_transform_bf is less efficient
        reverse_dist = dist.max() - dist
        reverse_dist = reverse_dist * gt_mask  # make outer background zero

        # CONSIDER EDGE CASES LIKE EMPTY AND FULL MASK
        if reverse_dist.max() == 0:
            if (
                gt_mask.max() == 0
            ):  # Grount-truth mask is empty (hiç bir pixelde etiket yok)
                # every element in dist is 0
                # IoU/Distance loss is defined for non-empty classes
                distance_weight = gt_mask  # full of 0s. It doesn't matter since it will be zerout out in loss func
                distance_weight_sum = np.sum(0)
                return distance_weight, distance_weight_sum
            elif (
                gt_mask.min() == 1
            ):  # Grount-truth mask is full (fotoğrafın her pixeli etiket)
                # every element in dist is -1
                distance_weight = gt_mask  # full of 1s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
            else:
                # If 1s in mask is really less. Ie: small objects. Eg: gt_mask.sum() = 13
                distance_weight = gt_mask  # very less 1s, too many 0s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum

        inner_reverse_dist_n = reverse_dist / reverse_dist.max()  # normalize it
        # pprint(f"Inner distance transform metrics: {inner_reverse_dist_n.min()}, {inner_reverse_dist_n.max()}, {inner_reverse_dist_n.mean()}")

        distance_weight_sum = np.sum(inner_reverse_dist_n)

        return inner_reverse_dist_n, distance_weight_sum

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        img = np.array(Image.open(os.path.join(self.root, path)).convert("RGB"))
        # If you need to convert to other format HWC -> CHW: img.transpose((-1, 0, 1))
        return img

    def _load_mask(self, id: int) -> List[Any]:
        height = self.coco.loadImgs(id)[0]["height"]
        width = self.coco.loadImgs(id)[0]["width"]
        if not self.coco.getAnnIds(id):  # check for empty masks
            arr_2dim = np.zeros((height, width)).astype(np.float32)
        else:
            # Non-empty mask case
            all_ann = self.coco.loadAnns(self.coco.getAnnIds(id))
            if len(all_ann) == 1: # if there is only single class in mask
                ann = all_ann[0]
                arr_2dim = self.coco.annToMask(ann=ann).astype(np.float32)
            elif len(all_ann) > 1: # if there is multi classes in mask
                arr_2dim = np.zeros((height, width, len(all_ann)), dtype=np.float32)
                for idx, sub_ann in enumerate(all_ann):
                    arr_2dim[:, :, idx] = self.coco.annToMask(ann=sub_ann).astype(np.float32)
        return arr_2dim  # in HW format

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        mask = self._load_mask(id)
        
        sample = dict(image=image, mask=mask)
        if self.transforms is not None:
            sample = self.transforms(**sample)
            # check for multiclass case
            class_num = sample["mask"].shape[2]  # HWC shape
            if class_num > 1:  # multiclass
                sample["distance_mask"] = torch.zeros_like(sample["mask"]).numpy()
                sample["distance_mask_sum"] = torch.zeros((class_num, 1)).numpy()
                for idx in range(class_num):
                    (
                        sample["distance_mask"][:, :, idx],
                        sample["distance_mask_sum"][idx],
                    ) = self._transform_binarymask_to_distance_mask(
                        np.array(sample["mask"][:, :, idx])
                    )  # change binary mask to distance transformed mask
                sample["distance_mask"] = sample["distance_mask"].transpose(
                    2, 0, 1
                )  # convert HWC to CHW format
                # import pdb; pdb.set_trace()
                sample["mask"] = np.array(sample["mask"]).transpose(
                    2, 0, 1
                )  # convert HWC to CHW format
            else:  # binary class
                (
                    sample["distance_mask"],
                    sample["distance_mask_sum"],
                ) = self._transform_binarymask_to_distance_mask(
                    np.array(sample["mask"])
                )  # change binary mask to distance transformed mask
                sample["distance_mask"] = np.expand_dims(
                    sample["distance_mask"], 0
                )  # convert to CHW format ie. HW -> 1HW:
                sample["mask"] = np.expand_dims(
                    sample["mask"], 0
                )  # convert to CHW format ie. HW -> 1HW:
        return sample
        
    def __len__(self) -> int:
        return len(self.ids)


# https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
class CityscapesToSmpDataset(Cityscapes):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, target_type="semantic")
        self.colormap = self._generate_19colormap()

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
        dist = ndimage.distance_transform_cdt(
            gt_mask
        )  # distance_transform_bf is less efficient
        reverse_dist = dist.max() - dist
        reverse_dist = reverse_dist * gt_mask  # make outer background zero

        # CONSIDER EDGE CASES LIKE EMPTY AND FULL MASK
        if reverse_dist.max() == 0:
            if (
                gt_mask.max() == 0
            ):  # Grount-truth mask is empty (hiç bir pixelde etiket yok)
                # every element in dist is 0
                # IoU/Distance loss is defined for non-empty classes
                distance_weight = gt_mask  # full of 0s. It doesn't matter since it will be zerout out in loss func
                distance_weight_sum = np.sum(0)
                return distance_weight, distance_weight_sum
            elif (
                gt_mask.min() == 1
            ):  # Grount-truth mask is full (fotoğrafın her pixeli etiket)
                # every element in dist is -1
                distance_weight = gt_mask  # full of 1s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
            else:
                # If 1s in mask is really less. Ie: small objects. Eg: gt_mask.sum() = 13
                distance_weight = gt_mask  # very less 1s, too many 0s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum

        inner_reverse_dist_n = reverse_dist / reverse_dist.max()  # normalize it
        # pprint(f"Inner distance transform metrics: {inner_reverse_dist_n.min()}, {inner_reverse_dist_n.max()}, {inner_reverse_dist_n.mean()}")

        distance_weight_sum = np.sum(inner_reverse_dist_n)

        return inner_reverse_dist_n, distance_weight_sum

    # https://github.com/albumentations-team/autoalbument/blob/master/examples/cityscapes/dataset.py
    def _generate_19colormap(self):
        # It gives 19 classes
        colormap = {}
        for class_ in self.classes:
            if class_.train_id in (-1, 255):
                continue
            colormap[class_.train_id] = class_.id
        return colormap

    def _convert_to_19segmentation_mask(self, mask):
        # It gives 19 classes
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(self.colormap)), dtype=np.float32
        )
        for label_index, label in self.colormap.items():
            segmentation_mask[:, :, label_index] = (mask == label).astype(float)
        return segmentation_mask

    def _generate_8colormap(self):
        # It gives 8 classes
        colormap = {}
        idx = 0
        for class_ in self.classes:
            if class_.category_id in (6, 7):
                if class_.name not in ("caravan", "trailer", "license plate"):
                    colormap[idx] = class_.id
                    idx += 1
        return colormap

    def _convert_to_8segmentation_mask(self, mask):
        # It gives 8 classes as in Mask-RCNN in this order:
        # person, rider, car, truck, bus, train, mcycle, bicycle
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, 8), dtype=np.float32)
        for label_index, label in self.colormap.items():
            segmentation_mask[:, :, label_index] = (mask == label).astype(float)
        return segmentation_mask

    def _generate_3colormap(self):
        # It gives 3 classes: person, bicycle, rider
        colormap = {}
        idx = 0
        for class_ in self.classes:
            if class_.category_id in (6, 7):
                if class_.name in ("person", "rider", "bicycle"):
                    colormap[idx] = class_.id
                    idx += 1
        return colormap

    def _convert_to_3segmentation_mask(self, mask):
        # It gives 3 classes as in Mask-RCNN in this order:
        # person, rider, car, truck, bus, train, mcycle, bicycle
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, 3), dtype=np.float32)
        for label_index, label in self.colormap.items():
            segmentation_mask[:, :, label_index] = (mask == label).astype(float)
        return segmentation_mask

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = np.array(Image.open(self.images[index]).convert("RGB"))

        # Just export semantic segmentation
        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "semantic":
                target = np.array(Image.open(self.targets[index][i])).astype(np.float32)
            else:
                print("target_type can only be 'semantic'")
                break
        mask = self._convert_to_19segmentation_mask(target)

        sample = dict(image=image, mask=mask)
        if self.transforms is not None:
            sample = self.transforms(**sample)
            # check for multiclass case
            class_num = sample["mask"].shape[2]  # HWC shape
            if class_num > 1:  # multiclass
                sample["distance_mask"] = torch.zeros_like(sample["mask"]).numpy()
                sample["distance_mask_sum"] = torch.zeros((class_num, 1)).numpy()
                for idx in range(class_num):
                    (
                        sample["distance_mask"][:, :, idx],
                        sample["distance_mask_sum"][idx],
                    ) = self._transform_binarymask_to_distance_mask(
                        np.array(sample["mask"][:, :, idx])
                    )  # change binary mask to distance transformed mask
                sample["distance_mask"] = sample["distance_mask"].transpose(
                    2, 0, 1
                )  # convert HWC to CHW format
                # import pdb; pdb.set_trace()
                sample["mask"] = np.array(sample["mask"]).transpose(
                    2, 0, 1
                )  # convert HWC to CHW format
            else:  # binary class
                (
                    sample["distance_mask"],
                    sample["distance_mask_sum"],
                ) = self._transform_binarymask_to_distance_mask(
                    np.array(sample["mask"])
                )  # change binary mask to distance transformed mask
                sample["distance_mask"] = np.expand_dims(
                    sample["distance_mask"], 0
                )  # convert to CHW format ie. HW -> 1HW:
                sample["mask"] = np.expand_dims(
                    sample["mask"], 0
                )  # convert to CHW format ie. HW -> 1HW:
        return sample


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
        dist = ndimage.distance_transform_cdt(
            gt_mask
        )  # distance_transform_bf is less efficient
        reverse_dist = dist.max() - dist
        reverse_dist = reverse_dist * gt_mask  # make outer background zero

        # CONSIDER EDGE CASES LIKE EMPTY AND FULL MASK
        if reverse_dist.max() == 0:
            if (
                gt_mask.max() == 0
            ):  # Grount-truth mask is empty (hiç bir pixelde etiket yok)
                # every element in dist is 0
                # IoU/Distance loss is defined for non-empty classes
                distance_weight = gt_mask  # full of 0s. It doesn't matter since it will be zerout out in loss func
                distance_weight_sum = np.sum(0)
                return distance_weight, distance_weight_sum
            elif (
                gt_mask.min() == 1
            ):  # Grount-truth mask is full (fotoğrafın her pixeli etiket)
                # every element in dist is -1
                distance_weight = gt_mask  # full of 1s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum
            else:
                # If 1s in mask is really less. Ie: small objects. Eg: gt_mask.sum() = 13
                distance_weight = gt_mask  # very less 1s, too many 0s
                distance_weight_sum = np.sum(distance_weight)
                return distance_weight, distance_weight_sum

        inner_reverse_dist_n = reverse_dist / reverse_dist.max()  # normalize it
        # pprint(f"Inner distance transform metrics: {inner_reverse_dist_n.min()}, {inner_reverse_dist_n.max()}, {inner_reverse_dist_n.mean()}")

        distance_weight_sum = np.sum(inner_reverse_dist_n)

        return inner_reverse_dist_n, distance_weight_sum

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transforms:
            sample = self.transforms(**sample)

        (
            sample["distance_mask"],
            sample["distance_mask_sum"],
        ) = self._transform_binarymask_to_distance_mask(
            np.array(sample["mask"])
        )  # change binary mask to distance transformed mask
        sample["distance_mask"] = np.expand_dims(
            sample["distance_mask"], 0
        )  # convert to CHW format ie. HW -> 1HW:
        sample["mask"] = np.expand_dims(
            sample["mask"], 0
        )  # convert to CHW format ie. HW -> 1HW:
        return sample

    def __len__(self):
        return len(self.subset)
