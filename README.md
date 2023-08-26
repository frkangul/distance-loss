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
The project requires Python 3.7.12 and several dependencies. You can set up the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```
## Configuration
The project uses YAML configuration files located in the config directory. You can specify the dataset, server, and experiment parameters in config.yaml. Dataset-specific parameters can be set in `dataset/<dataset_name>.yaml`, and server-specific parameters can be set in `server/<server_name>.yaml`.

## Training
To start the training process, run the train.py script:

```bash
python train.py
```

## Model
The main model class is ImageSegModel in `plModel.py`. It's a PyTorch Lightning module that encapsulates the model, loss function, optimizer, and learning rate scheduler.

## Custom Datasets
The `customDatasets.py` module contains custom dataset classes and utility functions for data manipulation. It includes a class for handling the MS Coco Detection dataset and a class for handling subsets of a dataset.

## Transforms
The `transforms.py` module defines image transformation functions using the Albumentations library.
