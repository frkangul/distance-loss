# LOAD MODEL FROM CKPT
model4eval = model # ImageSegModel.load_from_checkpoint("/kaggle/input/unet-sil/model_deneme")

# disable randomness, dropout, etc...
model4eval.eval()

test_ds = CocoToSmpDataset(root=os.path.join(cfg.dataset.data_dir, "test"), 
                                annFile=os.path.join(cfg.dataset.data_dir, "annotations_test.json"),
                                transforms=test_transform
                          )

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
    

un_img = np.array(test_ds[4]["image"]).transpose((1, 2, 0)) # CHW -> HWC
img = (unnorm_transform(image=un_img)["image"]*255).to(torch.uint8)

mask_gt = torch.tensor(test_ds[4]["mask"]) > 0.5 # boolean
show(draw_segmentation_masks(image=img, masks=mask_gt, alpha=0.5, colors=["orange"]))

# predict with the model
un_img = test_ds[4]["image"].unsqueeze(0)
y_hat = model4eval(un_img)
mask_pred = y_hat.sigmoid()
mask_pred = (mask_pred > 0.5).squeeze(0)
show(draw_segmentation_masks(image=img, masks=mask_pred, alpha=0.5, colors=["orange"]))


# LOAD MODEL FROM WANDB CKPT

# run = wandb.init(project="foreground-car-segm")

# reference can be retrieved in artifacts panel
# "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
# checkpoint_reference = "frkangul/foreground-car-segm/model-ugk2wlcc:v7"

# download checkpoint locally (if not already cached)
# artifact = run.use_artifact(checkpoint_reference, type="model")
# artifact_dir = artifact.download()

# load checkpoint
# model4eval = ImageSegModel.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

# trainer.test(model4eval)