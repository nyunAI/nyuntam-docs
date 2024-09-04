# Nyuntam Text Generation

## Overview

Nyuntam Text Generation evelops the SoTA compression methods and algorithms to achieve efficiency on text-generation tasks. This module implements model efficiency mixins via

- pruning
- quantization
- accelerations engines

## Basic Workflow

### Step 1 - Import a model and dataset

Check [Import Data](../dataset.md) and [Import Model](../model.md) for the exact steps. Make sure that the model-dataset combination provided is valid and of the same task.

### Step 2 - Choose an Algorithm

Check available [Algorithms](./algorithms.md) and respective hyperparameters. By default, most optimal hyperparameters are chosen, however, depending upon the task, model and dataset another set of hyperparameters can work better too.

### Step 3 - Monitor Logs and Export

By default, Job logs and model checkpoints are saved in nyuntam-text-generation/user_data/, however, users can specify the folder then want to store the logs in by changing the `LOGGING_PATH` argument in the yaml.