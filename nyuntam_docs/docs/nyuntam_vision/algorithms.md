# Algorithms

## Overview
nyuntam_vision currently supports the following tasks and respective algorithms -

* Image Classification -
    * [CPU Post Training Quantization - Torch](#cpu-post-training-quantization-torch)
    * [CPU Post Training Quantization - OpenVino](#cpu-post-training-quantization-openvino)
    * [CPU Post Training Quantization - ONNX](#cpu-post-training-quantization-onnx)
    * [GPU Post Training Quantization - TensorRT](#gpu-post-training-quantization-tensorrt)
    * [CPU Quantization Aware Training - Torch](#cpu-quantization-aware-training-torch)
    * [Knowledge Distillation](#knowledge-distillation)
    * [Structured Pruning](#structured-pruning-image-classification)
* Object Detection -
    * [CPU Post Training Quantization - ONNX](#cpu-post-training-quantization-onnx)
    * [GPU Post Training Quantization - TensorRT](#gpu-post-training-quantization-tensorrt)
    * [CPU Quantization Aware Training - OpenVino](#cpu-quantization-aware-training-openvino)
    * [Knowledge Distillation](#knowledge-distillation)
    * [Structured Pruning](#structured-pruning-object-detection)

### Vision Compressions
#### CPU Post Training Quantization - Torch
Native CPU quantization. 8 bit quantization by default. Outputs `.pt` model file which can be directly loaded by `torch.load`. 

| Parameter           | Values                   | Description                                      | Default Value  |
|----------------------|--------------------------|--------------------------------------------------|----------------|
| insize               | Int                | Input Shape For Vision Tasks (Currently only A X A Shapes supported) | 32             |
| BATCH_SIZE           | Int                | Batch Size                        | 1              |
| TRAINING             | bool               | Enables Finetuning before PTQ  | True           |
| VALIDATE             | bool               | Enables Validation during Optional Finetuning **(When TRAINING=True)** | True           |
| VALIDATION_INTERVAL  | Int                | Defines Epoch Intervals for Validation during Finetuning **(When TRAINING, VALIDATE = True)** | 1              |
| CRITERION            | "CrossEntropyLoss",<br>"MSE Loss",<br>[others](https://pytorch.org/docs/stable/nn.html#loss-functions) | Defines Loss functions for finetuning/validation **(When TRAINING = True)** | "CrossEntropyLoss" |
| LEARNING RATE        | Float           | Defines Learning Rate for Finetuning **(When TRAINING=True)** | 0.001          |
| FINETUNE_EPOCHS      | Int                | Defines the number of Epochs for Finetuning **(When TRAINING=True)** | 1              |
| OPTIMIZER            | "Adam",<br>"SGD",<br>[others](https://pytorch.org/docs/stable/optim.html#algorithms) | Defines Optimizer for Finetuning. **(When TRAINING = TRUE)** | Adam           |
| PRETRAINED           | bool               | Indicates whether to load ImageNet Weights in case custom model is not provided. | False          |          |
|choice                |"static","weight" or "fusion" | Indicates the Kind of PTQ to be performed. | "static"|


#### CPU Post Training Quantization - OpenVino
Neural networks inference optimization in OpenVINORuntime with minimal accuracy drop. Outputs `.xml` and `.bin` model files which can be directly loaded by `openvino.core.read_model`.

| Parameter           | Values                   | Description                                               | Default Value     |
|----------------------|--------------------------|-----------------------------------------------------------|-------------------|
| insize               | Int                | Input Shape For Vision Tasks (Currently only A X A Shapes supported) | 32                |
| BATCH_SIZE           | Int                | Batch Size                                  | 1                 |
| TRAINING             | bool               | Enables Finetuning before PTQ            | True              |
| VALIDATE             | bool               | Enables Validation during Optional Finetuning **(When TRAINING = True)** | True              |
| VALIDATION_INTERVAL  | Int                | Defines Epoch Intervals for Validation during Finetuning **(When TRAINING, VALIDATE = True)** | 1                 |
| CRITERION            | "CrossEntropyLoss",<br>"MSE Loss",<br>[others](https://pytorch.org/docs/stable/nn.html#loss-functions) | Defines Loss functions for finetuning/validation **(When TRAINING = True)** | CrossEntropyLoss  |
| LEARNING RATE        | Float          | Defines Learning Rate for Finetuning **(When TRAINING = True)**         | 0.001             |
| FINETUNE_EPOCHS      | Int                | Defines the number of Epochs for Finetuning **(When TRAINING = True)**      | 1                 |
| OPTIMIZER            | "Adam",<br>"SGD",<br>[others](https://pytorch.org/docs/stable/optim.html#algorithms) | Defines Optimizer for Finetuning. **(When TRAINING = TRUE)**           | Adam              |
| PRETRAINED           | bool               | Indicates whether to load ImageNet Weights in case custom model is not provided. | False             |
| TRANSFORMER           | bool               | Indicates whether uploaded model consists a transformer based architecture (Only For Classification) | True

#### CPU Post Training Quantization - ONNX
ONNX 8-bit CPU Post Training Quantization for Pytorch models. Outputs `.onnx` model files which can be directly loaded by `onnx.load`. 


|**Parameter**|**Values**|**Description**|**Default** **Value**|
| :- | :- | :- | :- |
|insize|Int|Input Shape For Vision Tasks (Currently only A X A Shapes supported) |32|
|BATCH\_SIZE|Int|Batch Size for dataloader|1|
|TRAINING|bool|Enables Finetuning before PTQ|True|
|VALIDATE|bool|<p>Enables Validation during Optional Finetuning</p><p>**(When TRAINING = True)**</p>|True|
|VALIDATION\_INTERVAL|Int|<p>Defines Epoch Intervals for Validation during Finetuning.</p><p>**(When TRAINING, VALIDATE = True)**</p>|1|
|CRITERION|<p>“CrossEntropyLoss”,</p><p>“MSE Loss”,</p><p>[others](https://pytorch.org/docs/stable/nn.html#loss-functions)</p>|<p>Defines Loss functions for finetuning/validation</p><p>**(When TRAINING = True)**</p>|CrossEntropyLoss|
|LEARNING RATE|Float|Defines Learning Rate for Finetuning **(When TRAINING = True)**|0\.001|
|FINETUNE\_EPOCHS|Int|<p>Defines the number of Epochs for Finetuning</p><p>**(When TRAINING = True)**</p>|1|
|OPTIMIZER|<p>“Adam”,</p><p>“SGD”,</p><p>[others](https://pytorch.org/docs/stable/optim.html#algorithms) </p>|Defines Optimizer for Finetuning. **(When TRAINING = True)**|Adam|
|PRETRAINED|bool|Indicates whether to load ImageNet Weights in case custom model is not provided.|False|
|quant\_format|QuantFormat.QDQ, QuantFormat.QOperator|Indicates the ONNX quantization representation format|QuantFormat.QDQ|
|per\_channel|bool|Indicates usage of "Per Channel" quantization that improves accuracy of models with large weight range|False|
|activation\_type|<p>QuantType.QInt8, QuantType.QUInt8, QuantType.QFLOAT8E4M3FN, QuantType.QInt16, QuantType.QUInt16</p><p></p>|Indicates the expected data type of activations post quantization|QuantType.QInt8|
|weight\_type|<p>QuantType.QInt8, QuantType.QUInt8, QuantType.QFLOAT8E4M3FN, QuantType.QInt16, QuantType.QUInt16</p><p></p>|Indicates the expected data type of weights post quantization|QuantType.QInt8|

#### CPU Quantization Aware Training - Torch
Native CPU quantization. 8 bit quantization by default. Outputs `.pt` model file which can be directly loaded by `torch.load`. 

| Parameter           | Values                   | Description                                      | Default Value  |
|----------------------|--------------------------|--------------------------------------------------|----------------|
| insize               | Int                | Input Shape For Vision Tasks (Currently only A X A Shapes supported) | 32             |
| BATCH_SIZE           | Int                | Batch Size                        | 1              |
| TRAINING             | bool               | Enables Finetuning before PTQ  | True           |
| VALIDATE             | bool               | Enables Validation during Optional Finetuning **(When TRAINING=True)** | True           |
| VALIDATION\_INTERVAL  | Int                | Defines Epoch Intervals for Validation during Finetuning **(When TRAINING, VALIDATE = True)** | 1              |
| CRITERION            | "CrossEntropyLoss",<br>"MSE Loss",<br>[others](https://pytorch.org/docs/stable/nn.html#loss-functions) | Defines Loss functions for finetuning/validation **(When TRAINING = True)** | "CrossEntropyLoss" |
| LEARNING RATE        | Float           | Defines Learning Rate for Finetuning **(When TRAINING=True)** | 0.001          |
| FINETUNE_EPOCHS      | Int                | Defines the number of Epochs for Finetuning **(When TRAINING=True)** | 1              |
| OPTIMIZER            | "Adam",<br>"SGD",<br>[others](https://pytorch.org/docs/stable/optim.html#algorithms) | Defines Optimizer for Finetuning. **(When TRAINING = TRUE)** | Adam           |
| PRETRAINED           | bool               | Indicates whether to load ImageNet Weights in case custom model is not provided. | False          |          |

#### CPU Quantization Aware Training - OpenVino
Neural networks inference optimization in OpenVINORuntime with minimal accuracy drop. Outputs `.xml` and `.bin` model files which can be directly loaded by `openvino.core.read_model`.

| Parameter           | Values                   | Description                                               | Default Value     |
|----------------------|--------------------------|-----------------------------------------------------------|-------------------|
| insize               | Int                | Input Shape For Vision Tasks (Currently only A X A Shapes supported) | 32                |
| BATCH_SIZE           | Int                | Batch Size                                  | 1                 |
| TRAINING             | bool               | Enables Finetuning before PTQ            | True              |
| VALIDATE             | bool               | Enables Validation during Optional Finetuning **(When TRAINING = True)** | True              |
| VALIDATION\_INTERVAL  | Int                | Defines Epoch Intervals for Validation during Finetuning **(When TRAINING, VALIDATE = True)** | 1                 |
| CRITERION            | "CrossEntropyLoss",<br>"MSE Loss",<br>[others](https://pytorch.org/docs/stable/nn.html#loss-functions) | Defines Loss functions for finetuning/validation **(When TRAINING = True)** | CrossEntropyLoss  |
| LEARNING RATE        | Float          | Defines Learning Rate for Finetuning **(When TRAINING = True)**         | 0.001             |
| FINETUNE_EPOCHS      | Int                | Defines the number of Epochs for Finetuning **(When TRAINING = True)**      | 1                 |
| OPTIMIZER            | "Adam",<br>"SGD",<br>[others](https://pytorch.org/docs/stable/optim.html#algorithms) | Defines Optimizer for Finetuning. **(When TRAINING = TRUE)**           | Adam              |
| PRETRAINED           | bool               | Indicates whether to load ImageNet Weights in case custom model is not provided. | False             |
|TRANSFORMER           | bool               | Indicates whether uploaded model consists a transformer based architecture | True


#### GPU Post Training Quantization - TensorRT
8-bit Quantization executable in GPU via TensorRT Runtime. Outputs `.engine` model file which can be directly loaded by `tensorrt.Runtime`. 

|**Parameter**|**Values**|**Description**|**Default** **Value**|
| :- | :- | :- | :- |
|insize|Int|Input Shape For Vision Tasks (Currently only A X A Shapes supported) |32|
|BATCH\_SIZE|Int|Batch Size for dataloader|1|
|TRAINING|bool|Enables Finetuning before PTQ|True|
|VALIDATE|bool|<p>Enables Validation during Optional Finetuning</p><p>**(When TRAINING = True)**</p>|True|
|VALIDATION\_INTERVAL|Int|<p>Defines Epoch Intervals for Validation during Finetuning.</p><p>**(When TRAINING, VALIDATE = True)**</p>|1|
|CRITERION|<p>“CrossEntropyLoss”,</p><p>“MSE Loss”,</p><p>[others](https://pytorch.org/docs/stable/nn.html#loss-functions)</p>|<p>Defines Loss functions for finetuning/validation</p><p>**(When TRAINING = True)**</p>|CrossEntropyLoss|
|LEARNING RATE|Float|Defines Learning Rate for Finetuning **(When TRAINING = True)**|0\.001|
|FINETUNE\_EPOCHS|Int|<p>Defines the number of Epochs for Finetuning</p><p>**(When TRAINING = True)**</p>|1|
|OPTIMIZER|<p>“Adam”,</p><p>“SGD”,</p><p>[others](https://pytorch.org/docs/stable/optim.html#algorithms) </p>|Defines Optimizer for Finetuning. **(When TRAINING = TRUE)**|Adam|
|PRETRAINED|bool|Indicates whether to load ImageNet Weights in case custom model is not provided.|False|


#### Knowledge Distillation
Simple Distillation Training Strategy that adds an additional loss between Teacher and Student Predictions. Outputs `.pt` model file which can be directly loaded by using `torch.load`.

|**Parameter**|**Values**|**Description**|**Default** **Value**|
| :- | :- | :- | :- |
|insize|Int|A single integer representing the input image size for teacher network and student network|32|
|BATCH\_SIZE|Int|Batch Size for dataloader|1|
|TRAINING|bool|Whether to finetune teacher model before distillation.|True|
|VALIDATE|bool|<p>Enables Validation during Optional Finetuning</p><p>**(When TRAINING = True)**</p>|True|
|VALIDATION\_INTERVAL|Int|<p>Defines Epoch Intervals for Validation during Finetuning.</p><p>**(When TRAINING, VALIDATE = True)**</p>|1|
|CRITERION|<p>“CrossEntropyLoss”,</p><p>“MSE Loss”,</p><p>[others](https://pytorch.org/docs/stable/nn.html#loss-functions)</p>|<p>Defines Loss functions for finetuning/validation</p><p>**(When TRAINING = True)**</p>|CrossEntropyLoss|
|LEARNING RATE|Float|Defines Learning Rate for Finetuning **(When TRAINING = True)**|0\.001|
|FINETUNE\_EPOCHS|Int|<p>Defines the number of Epochs for Finetuning</p><p>**(When TRAINING = True)**</p>|1|
|OPTIMIZER|<p>“Adam”,</p><p>“SGD”,</p><p>[others](https://pytorch.org/docs/stable/optim.html#algorithms) </p>|Defines Optimizer for Finetuning. **(When TRAINING = TRUE)**|Adam|
|TEACHER\_MODEL|String|Model Name of the provided Teacher Model. **(Required both when intrinsicly provided and when custom teacher is uploaded)**|vgg16|
|CUSTOM\_TEACHER\_PATH|String| Relative Path for Teacher checkpoint from User Data folder|None|
|METHOD| "pkd","cwd","pkd_yolo"| Distillation Algorithm to use to distill models.  Needed for MMDetection (pkd,cwd) and MMYolo(pkd_yolo) distillation. Not needed for classification.|pkd|
|EPOCHS|Int|Indicates Number of Training Epochs for Distillation|20|
|LR|Float|Indicates Learning Rate for distillation process.|0\.01|
|LAMBDA|Float|Adjusts the balance between cross entropy andKLDiv (Classification Only)|0\.5|
|TEMPERATURE|Int|Indicates Temperature for softmax (Classification Only)|20|
|SEED|Int|Sets the seed for random number generation (Classification Only)|43|
|WEIGHT\_DECAY|Float|Sets the amount of Weight Decay during Distillation (Classification Only) |0\.0005|

#### Structured Pruning (Image Classification)
Pruning existing Parameters to increase Efficiency. MM Detection and MM Segmentation models are currently supported through MM Razor Pruning Algorithms. Outputs `.pt` model file which can be directly loaded by `torch.load`. 

|**Parameter**|**Values**|**Description**|**Default** **Value**|
| :- | :- | :- | :- |
|insize|Int|Input Shape For Vision Tasks (Currently only A X A Shapes supported) |32|
|BATCH\_SIZE|Int|Batch Size for dataloader|1|
|TRAINING|bool|Whether to finetune model after pruning.|True|
|VALIDATE|bool|<p>Enables Validation during Optional Finetuning</p><p>**(When TRAINING = True)**</p>|True|
|VALIDATION\_INTERVAL|Int|<p>Defines Epoch Intervals for Validation during Finetuning.</p><p>**(When TRAINING, VALIDATE = True)**</p>|1|
|CRITERION|<p>“CrossEntropyLoss”,</p><p>“MSE Loss”,</p><p>[others](https://pytorch.org/docs/stable/nn.html#loss-functions)</p>|<p>Defines Loss functions for finetuning/validation</p><p>**(When TRAINING = True)**</p>|CrossEntropyLoss|
|LEARNING RATE|Float|Defines Learning Rate for Finetuning **(When TRAINING = True)**|0\.001|
|FINETUNE\_EPOCHS|Int|<p>Defines the number of Epochs for Finetuning</p><p>**(When TRAINING = True)**</p>|1|
|OPTIMIZER|<p>“Adam”,</p><p>“SGD”,</p><p>[others](https://pytorch.org/docs/stable/optim.html#algorithms) </p>|Defines Optimizer for Finetuning. **(When TRAINING = TRUE)**|Adam|
|PRETRAINED|bool|Indicates whether to load ImageNet Weights in case custom model is not provided.|False|

The below parameters are specifically for the pruning of classification models.
|**Parameter**|**Values**|**Description**|**Default** **Value**|
| :- | :- | :- | :- |
|PRUNER\_NAME|<p>MetaPruner,</p><p>GroupNormPruner,</p><p>BNSPruner,</p>|<p>Pruning Algorithm to be utilized for pruning classification models.</p>|GroupNormPruner|
|GROUP\_IMPORTANCE|<p>GroupNormImportance,</p><p>GroupTaylorImportance</p>|<p>Logic for identify importance of parameters to prune.</p>|<p>GroupNormImportance</p>|
|TARGET\_PRUNE\_RATE|Int|Parameter Reduction Rate, defines how much parameters are reduced|Integer Value|
|BOTTLENECK|bool|<p>When Pruning Transformer based Architectures, whether to prune only intermediate layers (bottleneck) or perform uniform pruning.</p>|<p>False</p>|
|PRUNE\_NUM\_HEADS|bool|Whether to Prune number of Heads (For Transformer based Architectures)|True|

#### Structured Pruning (Object Detection)
The below parameters are specifically for the pruning of object detection models.

|**Parameter**|**Values**|**Description**|**Default** **Value**|
| :- | :- | :- | :- |
|INTERVAL|Int|Epoch Interval between every pruning operation |10|
|NORM\_TYPE|"act", "flops"|Type of pruning operation. "act" focuses on reducing parameters with minimal changes to activations. "flops" focuses on improving number of flops.|"act"|
|LR\_RATIO|Float| Ratio to decrease lr rate.|0.1|
|TARGET\_FLOP\_RATIO|Float|The target flop ratio to prune your model. (also used for "act").|0.5|
|EPOCHS|Int| Number of epochs to perform training (possibly a multiple of Interval). |20|
