# A Distance Transform Based Loss Function for the Semantic Segmentation of Very High Resolution Remote Sensing Images

## Overview
This repository contains the source code and additional materials associated with the paper titled "A Distance Transform Based Loss Function for the Semantic Segmentation of Very High Resolution Remote Sensing Images," submitted to the IGARSS 2024 conference. This codebase contains the implementation of a distance loss algorithm for image segmentation tasks. It also includes training an image segmentation model. The project uses PyTorch Lightning for training and Weights & Biases for experiment tracking.

## Table of Contents
- [Environment Setup](#environment-etup)
- [Configuration](#configuration)
- [Data](#data)
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
## Configuration
The project uses YAML configuration files located in the config directory. You can specify the dataset, server, and experiment parameters in config.yaml. Dataset-specific parameters can be set in `config/dataset/<dataset_name>.yaml`, and server-specific parameters can be set in `config/server/<server_name>.yaml`.

Create a `.env` file in the root directory of the project. You need to connect to your WANDB accout. Generate WANDB API key from wandb website and add it into `.env` file as  `<your_wandb_api_key>`. See below:

```bash
WANDB=your_wandb_api_key
```

If there are too much wandb cache, then clean up:

```bash
wandb artifact cache cleanup 50GB
```

Generate a token on kaggle, then add this kaggle.json token into .kaggle folder.
## Data
Create a folder, named `data`, in the root directory of repository to store both CelebAMask-HQ (in coco format) and Cityscapes (it should contain `gtFine` and `leftImng8bit` folders) datasets. 
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
