# Distance Loss

This codebase contains the implementation of a distance loss algorithm for image segmentation tasks. I also includes training an image segmentation model. The project uses PyTorch Lightning for training and Weights & Biases for experiment tracking.

## Table of Contents
- [Environment Setup](#environment-etup)
- [Configuration](#configuration)
- [Training](#training)
- [Model](#model)
- [Custom Datasets](#custom-datasets)
- [Transforms](#transforms)

## Environment Setup
Firstly, check if there exists any GPU:

```bash
nvidia-smi
```

The project requires Python 3.7.12 and several dependencies. You can set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate distance-loss
```

You need to connect to your WANDB accout. Generate `<api-key>` from wandb website and add it into at the end of .bashrc file:

```bash
export WANDB=<api-key>
```

If there are too much wandb cache, then clean up:

```bash
wandb artifact cache cleanup 50GB
```

Generate a token on kaggle, then add this kaggle.json token into .kaggle folder.
## Configuration
The project uses YAML configuration files located in the config directory. You can specify the dataset, server, and experiment parameters in config.yaml. Dataset-specific parameters can be set in `config/dataset/<dataset_name>.yaml`, and server-specific parameters can be set in `config/server/<server_name>.yaml`.

Create a `.env` file in the root directory of the project with the following variables:
```bash
WANDB=your_wandb_api_key
```

Replace your_wandb_api_key with your actual WANDB API key.

## Training
To start the training process in the background and save logs:

```bash
nohup python train.py &
```
To see if the training process continues or not:

```bash
ps -ef | grep python
```
To kill the training process:

```bash
kill <pid>
```
## Model
The main model class is ImageSegModel in `plModel.py`. It's a PyTorch Lightning module that encapsulates the model, loss function, optimizer, and learning rate scheduler.

## Custom Datasets
The `customDatasets.py` module contains custom dataset classes and utility functions for data manipulation. It includes a class for handling the MS Coco Detection dataset and a class for handling subsets of a dataset.

## Transforms
The `transforms.py` module defines image transformation functions using the Albumentations library.
