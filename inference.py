import wandb
import os
from pathlib import Path
from src.models.plModel import ImageSegModel
from src.data.transforms import get_transforms
from src.data.customDatasets import CocoToSmpDataset, CityscapesToSmpDataset
import torch
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as visF
import matplotlib.pyplot as plt 
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

WANDB_KEY= os.environ["WANDB"]
wandb.login(key=WANDB_KEY)

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(50, 50))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = visF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        return fig, axs

@hydra.main(version_base=None, config_path="./config", config_name="config")
def inference_pipeline(cfg: DictConfig):
    run = wandb.init(project=cfg.exp.wandb_proj)

    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
    model_ref = "model-uysi6oe1"
    model_version = "v9"
    checkpoint_ref = f"frkangul/{run}/{model_ref}:{model_version}"

    # download checkpoint locally (if not already cached)
    artifact = run.use_artifact(checkpoint_ref, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    model = ImageSegModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    # disable randomness, dropout, etc...
    model.eval()
    
    transforms = get_transforms(cfg)

    if cfg.dataset.name == "CelebAMask-HQ":
        test_ds = CocoToSmpDataset(root=os.path.join(cfg.dataset.dir, "test"),
                                   annFile=os.path.join(cfg.dataset.dir, "annotations_test.json"),
                                   transforms=transforms["test"])
    elif cfg.dataset.name == "Cityscapes":
        test_ds = CityscapesToSmpDataset(cfg.dataset.dir, split='test',
                                         mode='fine', transforms=transforms["test"])

    random_img_id = 4
    threshold = 0.5
    
    un_img = np.array(test_ds[random_img_id]["image"]).transpose((1, 2, 0)) # CHW -> HWC
    img = (transforms["unnorm"](image=un_img)["image"]*255).to(torch.uint8)

    # show GT mask
    mask_gt = torch.tensor(test_ds[random_img_id]["mask"]) > 0.5 # boolean
    show(draw_segmentation_masks(image=img, masks=mask_gt, alpha=0.5, colors=["orange"]))

    # show prediction with the model
    un_img = test_ds[random_img_id]["image"].unsqueeze(0)
    y_hat = model(un_img)
    mask_pred = y_hat.sigmoid()
    mask_pred = (mask_pred > threshold).squeeze(0)
    show(draw_segmentation_masks(image=img, masks=mask_pred, alpha=0.5, colors=["orange"]))


if __name__ == "__main__":
    inference_pipeline()