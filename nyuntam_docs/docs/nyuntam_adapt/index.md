# Adapt

## Overview

Adapt is a no-code toolkit that enables Parameter-Efficient Fine-Tuning (PEFT) techniques such as LoRA, DoRA, QLoRA, and SSF for AI model fine-tuning across various tasks, including language and vision. Designed as a simple no-code platform, it simplifies the application of PEFT to enhance model performance for tasks like text classification, generation, summarization, question answering, image classification, object detection, and segmentation. Adapt democratizes advanced AI customization, making it accessible to a broader audience.

## Basic Workflow

### Step 1 - Import a model and dataset.

Check [Import Data](../dataset.md) and [Import Model](../model.md) for the exact steps. Make sure that the model-dataset combination provided is valid and of the same task.

### Step 2 - Choose an Algorithm.

Check available [Algorithms](./algorithms.md) and respective hyperparameters. By default, most optimal hyperparameters are chosen, however, depending upon the task, model and dataset another set of hyperparameters can work better too.

### Step 3 - Monitor Logs and Export.

By default, Job logs and model checkpoints are saved in Adapt/user_data/, however, users can specify the folder then want to store the logs in by changing the `LOGGING_PATH` argument in the yaml.