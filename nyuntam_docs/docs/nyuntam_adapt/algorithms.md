# Algorithms

## Overview 
nyuntam_adapt currently supports the following tasks - 

* [Text Generation](#text-generation)
* [Text Classification](#text-classification)
* [Summarization](#summarization)
* [Translation](#translation)
* [Question Answering](#question-answering)
* [Image Classification](#image-classification)
* [Object Detection](#object-detection)
* [Instance Segmentation](#instance-segmentation)
* [Pose Detection](#pose-detection)

The following techniques are supported to adapt any model for the above mentioned tasks - 

* [LoRA](#lora-low-rank-adaptation)
* [SSF](#ssf-scaling-and-shifting-your-features)
* [DoRA](#dora-weight-decomposed-low-rank-adaptation)
* [QLoRA (4-bit and 8-bit)](#qlora-quantized-low-rank-adaptation)
* [QDoRA (4-bit and 8-bit)](#qdora-weight-decomposed-low-rank-adaptation)
* [QSSF (4-bit)](#q-ssf-quantized-ssf)


### LoRA (Low Rank Adaptation) 
LoRA is a Parameter-Efficient Fine-Tuning (PEFT) technique that optimizes AI models by introducing low-rank matrices to adjust the weights of pre-trained neural networks. This approach allows for significant improvements in model performance with minimal additional parameters, making it an efficient method for customizing AI models for specific tasks without the need for extensive retraining or computational resources.

Argument : ***method = "LoRA"***

| Parameter            | Value (Datatype) | Default Value | Description                                    |
|----------------------|------------------|---------------|------------------------------------------------|
| `r`                    |  int           |      8         | Rank of the low-rank approximation.             |
| `alpha`                | int      |        16       | Scaling factor for LoRA adjustments.           |
| `dropout`              | float      |       0.1        | Dropout rate for regularization.               |
| `target_modules`       |                  |               | Modules within the model targeted for tuning.  |
| `fan_in_fan_out`       | bool      |      False         | Whether to adjust initialization based on fan-in/fan-out. |
| `init_lora_weights`    | bool    |       True        | Initializes LoRA weights if set to True.       |

### SSF (Scaling and Shifting your Features) 
SSF adjusts the scale (multiplication) and shift (addition) of features within a neural network to better adapt the model to specific tasks or datasets. By applying these simple yet effective transformations, SSF aims to enhance model performance without the need for extensive retraining or adding a significant number of parameters. This technique is particularly useful for fine-tuning pre-trained models in a more resource-efficient manner, allowing for targeted improvements with minimal computational cost.

Argument : ***method = "SSF"***

### DoRA (Weight-Decomposed Low-Rank Adaptation)
DoRA decomposes the pre-trained weight into two components, magnitude and direction, for fine-tuning, specifically employing LoRA for directional updates to efficiently minimize the number of trainable parameters. DoRA enhances both the learning capacity and training stability of LoRA while avoiding any additional inference overhead.

| Parameter       | Datatype | Default Value | Description          |
|-----------------|----------|---------------|----------------------|
| `r`               | integer  | 16            |Rank of the low-rank approximation.                       |
| `alpha`           | integer  | 8             |Scaling factor for LoRA adjustments.                      |
| `dropout`         | float    | 0.1           |Dropout rate for regularization.                           |
| `target_modules`  | list     |               |The names of the layers for which peft modules will be created                     |
| `fan_in_fan_out`  | boolean  | False         |Initializes DoRA weights if set to True.                      |

Argument : ***method = "DoRA"***

### QLoRA (Quantized Low Rank Adaptation) 
QLoRA (Quantized Low-Rank Adaptation) is a sophisticated Parameter-Efficient Fine-Tuning (PEFT) technique designed to enhance the adaptability and efficiency of pre-trained AI models with minimal computational overhead. The weight matrices of the original model are frozen and quantized thus reducing the memory footprint of the original model. QLoRA is particularly useful in scenarios where computational resources are limited, offering a balance between model adaptability, performance, and resource efficiency.

Arguments : <ul>
     <li>***method = "LoRA"***</li>
     <li>***load_in_4bit = "True" (for 4 bit quantization)***</li>
     <li>***load_in_8bit = "True" (for 8 bit quantization)***</li></ul>
    
### QDoRA (Weight-Decomposed Low-Rank Adaptation) 
QDoRA (Weight-Decomposed Low-Rank Adaptation) is a sophisticated Parameter-Efficient Fine-Tuning (PEFT) technique designed to enhance the adaptability and efficiency of pre-trained AI models with minimal computational overhead.  QDoRA is particularly useful in scenarios where computational resources are limited, offering a balance between model adaptability, performance, and resource efficiency.

Arguments : <ul>
     <li>***method = "DoRA"***</li>
     <li>***load_in_4bit = "True" (for 4 bit quantization)***</li>
     <li>***load_in_8bit = "True" (for 8 bit quantization)***</li></ul>



### Q-SSF (Quantized SSF)
 Quantized SSF applies quantizes the frozen model weights. The SSF modules are trained, enhancing model efficiency with minimal fidelity loss. This approach reduces memory and computational demands, ideal for resource-constrained environments, maintaining accuracy while improving performance.

Arguments : <ul>
     <li>***method = "SSF"***</li>
     <li>***load_in_4bit = "True" (for 4 bit quantization)***</li>
     <li>***load_in_8bit = "True" (for 8 bit quantization)***</li></ul>

#### Quantization Parameters for QLoRA/QDoRA/Q-SSF
Model Quantization is achieved via BitsandBytes module.

**4 bit Quantization**

| Parameter                     | Datatype | Default Value | Description                                      |
|-------------------------------|----------|---------------|--------------------------------------------------|
| `load_in_4bit`                  | bool     | True          | Load model in 4-bit precision.                   |
| `bnb_4bit_compute_dtype`        | str      | 'float16'     | Compute data type in 4-bit mode.                |
| `bnb_4bit_quant_type`           | str      | 'nf4'         | Quantization type for 4-bit precision.          |
| `bnb_4bit_use_double_quant`     | bool     | False         | Use double quantization in 4-bit mode.          |

**8 bit Quantization**

| Parameter                           | Datatype | Default Value | Description                                                     |
|-------------------------------------|----------|---------------|-----------------------------------------------------------------|
| `load_in_8bit`                        | bool     | False         | Load model in 8-bit precision (only for float16/bfloat16 weights). |
| `llm_int8_threshold`                  | float    | 6.0           | Threshold for LLM int8 quantization.                            |
| `llm_int8_skip_modules`               |          |               | Modules to skip during LLM int8 quantization.                   |
| `llm_int8_enable_fp32_cpu_offload`    | bool     | False         | Enable FP32 offload to CPU in int8 mode.                        |
| `llm_int8_has_fp16_weight`            | bool     | False         | Whether the model has fp16/bfloat16 weights                     |


### Full Fine-tuning
This method updates all parameters of a neural network. This method is generally inefficient to run as it uses a lot of computing power to update all parameters of the given model. 

Arguments : <ul>
     <li>***FULL_FINE_TUNING = "True"***</li>
     </ul>

### Last Layer Tuning
Tuning the last layer is a popular technique in which all the layers of a model, except the last few layers are frozen i.e. their gradients are not updated during training. This method is generally used while finetuning a model trained on a huge generalized dataset (Imagenet) for downstream tasks. 
Arguments : <ul>
     <li>***LAST_LAYER_TUNING = "True"***</li>
     </ul>

***NOTE** :<ul><li>[Full Fine Tuning](#full-fine-tuning) should be used without any PEFT methods.</li><li>[Last Layer Tuning](#last-layer-tuning) should be set to "True" while using any PEFT method.</li> </ul>

## Common Training Parameters 

The wide variety of tasks and algorithms supported in nyuntam_adapt have several parameters that are common across all tasks/algorithms and some parameters that are specific to a particular task/algorithm

| Parameter                   | Datatype | Default Value | Description                                     |
|-----------------------------|----------|---------------|-------------------------------------------------|
| `DO_TRAIN`                    | bool     | true          | Flag indicating whether to perform training    |
| `DO_EVAL`                     | bool     | true          | Flag indicating whether to perform evaluation |
| `NUM_WORKERS`                 | int      | 4             | Number of worker processes for data loading    |
| `BATCH_SIZE`                  | int      | 16            | Batch size for training                        |
| `EPOCHS`                      | int      | 2             | Number of epochs for training                  |
| `OPTIMIZER`                   | str      | adamw_torch   | Optimization algorithm (sgd,adamw_torch,paged_adamw_32bit)       |
| `LR`                          | float    | 1e-4          | Learning rate                                  |
| `SCHEDULER_TYPE`              | str      | linear        | Type of learning rate scheduler(linear,constant) **Supported Schedulers :** <ul><li>Object deteciton, Instance Segmentation, Pose Detection - <ul><li>CosineAnnealingLR</li> <li>LinearLR</li> <li>MultiStepLR</li> </li></ul><li> NLP Tasks, Image Classification - All huggingface schedulers           |
| `WEIGHT_DECAY`                | float    | 0.0           | Weight decay for optimization                 |
| `BETA1`                       | float    | 0.9           | Beta1 parameter for Adam optimizer             |
| `BETA2`                       | float    | 0.999         | Beta2 parameter for Adam optimizer             |
| `ADAM_EPS`                    | float    | 1e-8          | Epsilon value for Adam optimizer               |
| `INTERVAL`                    | str      | epoch         | Interval type for checkpointing (e.g., epoch)  |
| `INTERVAL_STEPS`              | int      | 100           | Steps interval for checkpointing              |
| `NO_OF_CHECKPOINTS`           | int      | 5             | Number of checkpoints to save during training |
| `FP16`                        | bool     | false         | Flag indicating whether to use FP16 precision  |
| `RESUME_FROM_CHECKPOINT`      | bool     | false         | Flag indicating whether to resume from a checkpoint |
| `GRADIENT_ACCUMULATION_STEPS` | int      | 1             | Number of steps to accumulate gradients       |
| `GRADIENT_CHECKPOINTING`      | bool     | false         | Flag indicating whether to use gradient checkpointing |
| `SAVE_METHOD`           | string     | 'state_dict'| The method in which the model will be saved (Values - 'full_torch_model' : Saves the model as a .pt file in full precision, 'state_dict' : Saves the model state dictionary,  'safetensors' : Saves the model weights as safetensors (Advisable for huggingface models) ,'save_pretrained':saves the model as a folder using huggingface's save_pretrained method (Only supporte for huggingface models.)  ) |


## DDP arguments

DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines. The recommended way to run adaption for a model with DDP is to use the following parameters :

| Parameter            | Data Type  | Default Value               | Description                                                      |
|----------------------|------------|-----------------------------|------------------------------------------------------------------|
| `DDP`              | bool    | True                        | A boolean indicating whether DDP is enabled or not.         |
| `num_nodes`   | int    | 1                      | An integer refering to the number of nodes used for DDP training             |

## FSDP arguments
Training AI models at a large scale is a challenging task that requires a lot of compute power and resources. It also comes with considerable engineering complexity to handle the training of these very large models. To address this Fully Sharded Data Parallel (FSDP) comes inbuilt with nyuntam-adapt. When training with FSDP, the GPU memory footprint is smaller than when training with DDP across all workers. This makes the training of some very large models feasible by allowing larger models or batch sizes to fit on device. This comes with the cost of increased communication volume. The communication overhead is reduced by internal optimizations like overlapping communication and computation.

| Parameter Name                        | Data Type  | Default Value | Description                                      |
|---------------------------------------|------------|---------------|--------------------------------------------------|
| `FSDP`                 | Bool     | True | Boolean indicating whether FSDP is enabled or not. |
| `compute_environment`                 | String     | LOCAL_MACHINE | Specifies the environment where the computation runs. |
| `debug`                               | Boolean    | false         | Indicates whether debugging is enabled.          |
| `distributed_type`                    | String     | FSDP          | Defines the type of distributed training.        |
| `downcast_bf16`                       | String     | 'no'          | Controls downcasting to bf16 precision.          |
| `fsdp_auto_wrap_policy`               | String     | TRANSFORMER_BASED_WRAP | Auto-wrap policy for FSDP.             |
| `fsdp_backward_prefetch_policy`       | String     | BACKWARD_PRE  | Prefetch policy during FSDP backward pass.       |
| `fsdp_forward_prefetch`               | Boolean    | false         | Indicates if forward prefetching is enabled.     |
| `fsdp_cpu_ram_efficient_loading`      | Boolean    | true          | Enables efficient loading into CPU RAM.          |
| `fsdp_offload_params`                 | Boolean    | false         | Indicates if parameters are offloaded.           |
| `fsdp_sharding_strategy`              | String     | FULL_SHARD    | Defines the FSDP sharding strategy.              |
| `fsdp_state_dict_type`                | String     | SHARDED_STATE_DICT | Type of state dictionary used in FSDP.      |
| `fsdp_sync_module_states`             | Boolean    | true          | Synchronizes module states across processes.     |
| `fsdp_use_orig_params`                | Boolean    | true          | Indicates if original parameters are used.       |
| `machine_rank`                        | Integer    | 0             | Rank of the machine in the distributed setup.    |
| `main_training_function`              | String     | main          | Name of the main training function to run.       |
| `mixed_precision`                     | String     | bf16          | Specifies the mixed precision type.              |
| `num_machines`                        | Integer    | 1             | Number of machines used in training.             |
| `num_processes`                       | Integer    | 2             | Number of processes per machine.                 |
| `rdzv_backend`                        | String     | static        | Rendezvous backend used for distributed training.|
| `same_network`                        | Boolean    | true          | Indicates if machines are on the same network.   |
| `tpu_env`                             | Array      | []            | Environment variables for TPU.                   |
| `tpu_use_cluster`                     | Boolean    | false         | Indicates if TPU cluster usage is enabled.       |
| `tpu_use_sudo`                        | Boolean    | false         | Indicates if sudo is required for TPU usage.     |
| `use_cpu`                             | Boolean    | false         | Indicates if CPU is used instead of GPU/TPU.     |




## Text Generation

| Parameter            | Data Type  | Default Value               | Description                                                      |
|----------------------|------------|-----------------------------|------------------------------------------------------------------|
| `packing`              | bool    | True                        | A boolean indicating whether packing is enabled or not.         |
| `dataset_text_field`   | str    | 'text'                      | The field in the dataset containing the text data.              |
| `max_seq_length`       | int   | 512                         | The maximum sequence length allowed for input text.             |
| `flash_attention2`       | bool   | Fasle                         | Argument to indicate whether to use flash attention or not (Warning - most of the models don't support flash attention which might lead to unexpected behaviours)|




## Text Classification

nyuntam_adapt supports <ul>
     <li> **Token Classification** : <ul>
               <li> **Named entity recognition (NER)** : Find the entities (such as persons, locations, or organizations) in a sentence. This can be formulated as attributing a label to each token by having one class per entity and one class for “no entity.” </li>
               <li> **Part-of-speech tagging (POS)**: Mark each word in a sentence as corresponding to a particular part of speech (such as noun, verb, adjective, etc.). </li>
               <li> **Chunking**: Find the tokens that belong to the same entity. This task (which can be combined with POS or NER) can be formulated as attributing one label (usually B-) to any tokens that are at the beginning of a chunk, another label (usually I-) to tokens that are inside a chunk, and a third label (usually O) to tokens that don’t belong to any chunk.</li>
               </ul>
     <li> **Text Classification** : Classification of given text into 2 or more classes (examples - emotion recognition)</li>
     </ul>


| Parameter        | Data Type  | Default Value            | Description                                      |
|------------------|------------|--------------------------|--------------------------------------------------|
| `subtask`          | str     | None                    | The specific subtask associated with the model ("ner" ,"pos", "chunk", ). If subtask = None, then the task is classic text classification|

## Summarization

| Parameter               | Data Type  | Default Value | Description                                       |
|-------------------------|------------|---------------|---------------------------------------------------|
| `MAX_TRAIN_SAMPLES`     | int      |      1000               | Maximum number of training samples           |
| `MAX_EVAL_SAMPLES`      | int       |        1000            | Maximum number of evaluation samples        |
| `max_input_length`        | int    | 512           | The maximum length allowed for input documents.    |
| `max_target_length`       | int    | 128           | The maximum length allowed for generated summaries.|
| `eval_metric`             | str    | 'rouge'       | The evaluation metric used during training and evaluation (options: 'bleu', 'rouge'). |
| `generation_max_length`   | int    | 128           | The maximum length allowed for generated text during prediction. |


## Translation

| Parameter            | Data Type  | Default Value             | Description                                       |
|----------------------|------------|---------------------------|---------------------------------------------------|
| `max_input_length`     | int    | 128                       | The maximum length allowed for input sentences.    |
| `max_target_length`    | int    | 128                       | The maximum length allowed for translated sentences.|
| `eval_metric`          | str    | 'rouge'                   | The evaluation metric used during training and evaluation (options: 'sacrebleu', 'rouge'). |
| `source_lang`          | str     | 'en'                      | The source language for translation (e.g., English). |
| `target_lang`          | str     | 'ro'                      | The target language for translation (e.g., Romanian). |
| `PREFIX`          | str     | example -'translate English to Russian: '                    | For multi-task models like t5, prefix is attached during specific tasks |



## Question Answering

nyuntam_adapt support span detection in question answering.

| Parameter            | Data Type  | Default Value             | Description                                       |
|----------------------|------------|---------------------------|---------------------------------------------------|
| `MAX_TRAIN_SAMPLES`     | int      |      1000               | Maximum number of training samples           |
| `MAX_EVAL_SAMPLES`      | int       |        1000            | Maximum number of evaluation samples        |
| `max_answer_length`    | int    | 30                        | The maximum length allowed for the generated answers. |
| `max_length`           | int    | 384                       | The maximum length allowed for input documents.    |
| `doc_stride`           | int   | 128                       | The stride used when the context is too large and is split across several features|


## Image Classification

| Parameter               | Data Type  | Default Value                | Description                                       |
|-------------------------|------------|------------------------------|---------------------------------------------------|
| `load_model`             | bool    | False                        | A boolean indicating whether to load a pre-trained model. |
| `model_path`             | str     | "densenet121"                | The path or identifier of the pre-trained model to be loaded. |
| `model_type`             | str     | 'densenet_timm'              | The type of the loaded model (e.g., 'densenet_timm'). |
| `image_processor_path`   | str     | 'facebook/convnext-tiny-224' | The path or identifier of the image processor configuration. |


## Object Detection



| Parameter                   | Datatype | Default Value | Description                                     |
|-----------------------------|----------|---------------|-------------------------------------------------|
| `BEGIN`                        | int     | 0         | The epoch from which the learning rate scheduler starts  |
| `END`                        | int     | 50         | The epoch at which the learning rate scheduler stops  |
| `T_MAX`                        | int     | 0         | Maximum number of iterations. (Exclusive Parameter for CossineAnnealingLR scheduler)  |
| `WARMUP`                        | bool     | false         | Flag to indicate whether to use wamrup iters or not|
| `WARMUP_RATIO`                        | float     | 0.1         | The ratio of (wamrup learning rate)/(real learning rate) |
| `WARMUP_ITERS`                        | int     | 50         | The number of warmup iterations |
| `MILESTONES`                        | list     | []         | List of epoch indices in increagin order where the LR changes (exclusive parameter for MultiStepLR scheduler) |
| `GAMMA`                        | float     | 0.1         | Multiplicative factor of learning rate decay.|
| `amp`                  | bool            | False         | Automatic mixed precision training.                     |
| `auto_scale_lr`        | bool            | False         | Enable automatic scaling of learning rates.             |
| `cfg_options`          | bool or dict    | None          | Additional configuration options for the MMDET model. If True, it indicates the default options should be used. |
| `train_ann_file`       | str             | 'train.json'  | Annotation file for training in COCO format. |
| `val_ann_file`         | str             | 'val.json'    | Annotation file for validation in COCO format. |
| `checkpoint_interval`  | int             | 5             | Interval for saving checkpoints during training (in epochs). |

## Instance Segmentation

| Parameter                   | Datatype | Default Value | Description                                     |
|-----------------------------|----------|---------------|-------------------------------------------------|
| `BEGIN`                        | int     | 0         | The epoch from which the learning rate scheduler starts  |
| `END`                        | int     | 50         | The epoch at which the learning rate scheduler stops  |
| `T_MAX`                        | int     | 0         | Maximum number of iterations. (Exclusive Parameter for CossineAnnealingLR scheduler)  |
| `WARMUP`                        | bool     | false         | Flag to indicate whether to use wamrup iters or not|
| `WARMUP_RATIO`                        | float     | 0.1         | The ratio of (wamrup learning rate)/(real learning rate) |
| `WARMUP_ITERS`                        | int     | 50         | The number of warmup iterations |
| `MILESTONES`                        | list     | []         | List of epoch indices in increagin order where the LR changes (exclusive parameter for MultiStepLR scheduler) |
| `GAMMA`                        | float     | 0.1         | Multiplicative factor of learning rate decay.|
| `amp`                  | bool            | False         | Automatic mixed precision training.                     |
| `auto_scale_lr`        | bool            | False         | Enable automatic scaling of learning rates.             |
| `cfg_options`          | bool or dict    | None          | Additional configuration options for the MMDET model. If True, it indicates the default options should be used. |
| `train_ann_file`       | str             | 'train.txt'  | Annotation file for training containing all the image names in train folder (without the extension)|
| `val_ann_file`         | str             | 'val.txt'    |Annotation file for training containing all the image names in test folder (without the extension)|
| `checkpoint_interval`  | int             | 5             | Interval for saving checkpoints during training (in epochs). |
| `class_list`  | list             | []             | List containing all the classes in the segmentation task|
| `palette`  | list             | []             | List of Lists containing the RGB value for each class|


## Pose Detection

| Parameter                   | Datatype | Default Value | Description                                     |
|-----------------------------|----------|---------------|-------------------------------------------------|
| `BEGIN`                        | int     | 0         | The epoch from which the learning rate scheduler starts  |
| `END`                        | int     | 50         | The epoch at which the learning rate scheduler stops  |
| `T_MAX`                        | int     | 0         | Maximum number of iterations. (Exclusive Parameter for CossineAnnealingLR scheduler)  |
| `WARMUP`                        | bool     | false         | Flag to indicate whether to use wamrup iters or not|
| `WARMUP_RATIO`                        | float     | 0.1         | The ratio of (wamrup learning rate)/(real learning rate) |
| `WARMUP_ITERS`                        | int     | 50         | The number of warmup iterations |
| `MILESTONES`                        | list     | []         | List of epoch indices in increagin order where the LR changes (exclusive parameter for MultiStepLR scheduler) |
| `GAMMA`                        | float     | 0.1         | Multiplicative factor of learning rate decay.|
| `amp`                  | bool            | False         | Automatic mixed precision training.                     |
| `auto_scale_lr`        | bool            | False         | Enable automatic scaling of learning rates.             |
| `cfg_options`          | bool or dict    | None          | Additional configuration options for the MMDET model. If True, it indicates the default options should be used. |
| `train_ann_file`       | str             | 'annotations/person_keypoints_val2017.json'  | Annotation file for training in COCO-Pose format. |
| `val_ann_file`         | str             | 'annotations/person_keypoints_val2017.json'    | Annotation file for validation in COCO-Pose format. |
| `checkpoint_interval`  | int             | 5             | Interval for saving checkpoints during training (in epochs). |

