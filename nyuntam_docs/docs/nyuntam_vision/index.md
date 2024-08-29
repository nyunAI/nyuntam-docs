# Nyuntam Vision

## Overview

Developed to compress and optimize deep learning models, Nyuntam Vision provides a set of compression techniques tailored for specific deployment constraints. Users have the flexibility to choose and combine multiple techniques to achieve the best trade-off between model performance and deployment constraints. Leveraging cutting-edge techniques like pruning, quantization, distillation, etc. It achieves exceptional model compression levels on a variety of vision models.

## Basic Workflow

### Step 1 - Import a model and dataset

Check [Import Data](../dataset.md) for the exact steps. Make sure that the model-dataset combination provided is valid and of the same task.

### Step 2 - Choose an Algorithm

Check available [Algorithms](./algorithms.md) and respective hyperparameters. By default, most optimal hyperparameters are chosen, however, depending upon the task, model and dataset another set of hyperparameters can work better too.

### Step 3 - Monitor Logs and Export

By default, Job logs and model checkpoints are saved in nyuntam_vision/user_data/, however, users can view the logs and download their compressed models locally using the export functionality.
