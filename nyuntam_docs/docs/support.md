# Model Compression & Adaptation Support Grid

Below are tables summarizing the support for various compression and adaptation techniques across different models. Apart from these, other similar models may be supported but have not been tested.

## Nyuntam Vision

Note that in the table below, CPU and GPU indicate the target device of deployment. PTQ indicates Post Training Quantization, and QAT indicates Quantization Aware Training.

| Model              | CPU PTQ - Torch | CPU PTQ - OpenVINO | CPU PTQ - ONNX | GPU PTQ - TensorRT | CPU QAT - Torch | Knowledge Distillation | Structured Pruning | CPU QAT - OpenVINO |
|--------------------|-----------------|--------------------|----------------|--------------------|-----------------|------------------------|--------------------|--------------------|
| ResNet (timm)      | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| ConvNextV2 (huggingface) | -               | ✓                  | ✓              | ✓                  | -               | ✓                      | ✓                  | ✓                  |
| MobileNetV3 (timm)  | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| DeiT (huggingface)  | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| VanillaNet (timm)   | -               | -                  | ✓              | ✓                  | -               | ✓                      | ✓                  | ✓                  |
| Swin (huggingface)  | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| YoloX (mmyolo/mmdet)| -               | ✓                  | -              | ✓                  | ✓               | ✓                      | ✓                  | -                  |
| RTMDet (mmyolo/mmdet)| -              | ✓                  | -              | ✓                  | ✓               | ✓                      | -                  | -                  |
| YOLOv8 (mmyolo)     | -               | -                  | -              | ✓                  | -               | ✓                      | -                  | -                  |

## Nyuntam Text-Generation

| Model   	| AWQ 	| LMQuant (QoQ) 	| AQLM 	| TensorRT 	| Exllama 	| MLC-LLM 	| FLAP 	|
|---------	|-----	|---------------	|------	|----------	|---------	|---------	|------	|
| LLaMA   	| ✓   	| ✓             	| ✓    	| ✓        	| ✓       	| ✓       	| ✓    	|
| LLaMA-2 	| ✓   	| ✓             	| ✓    	| ✓        	| ✓       	| ✓       	| ✓    	|
| LLaMA-3 	| ✓   	| ✓             	| ✓    	| -        	| ✓       	| -       	| ✓    	|
| Vicuna  	| ✓   	| ✓             	| ✓    	| ✓        	| ✓       	| ✓       	| ✓    	|
| Mistral 	| ✓   	| ✓             	| ✓    	| ✓        	| ✓       	| ✓       	| ✓    	|
| Mixtral 	| ✓   	| ✓             	| ✓    	| ✓        	| ✓       	| ✓       	| ✓    	|
| Gemma   	| ✓   	| -             	| ✓    	| ✓        	| -       	| -       	| -    	|

## Nyuntam Adapt

### LLM Tasks 
- Text Generation 
- Summarization 
- Question Answering 
- Text Classification 
- Translation 

**All of the major Huggingface models are supported for these tasks.**

### Image Classification

**All major image models on Huggingface and timm are supported in Adapt.**

### Object Detection

| Model  | LoRA | SSF  | DoRA | Full Fine Tuning |
|--------|------|------|------|------------------|
| YoloX  | ✓    | ✓    | ✓    | ✓                |
| RTMDet | ✓    | ✓    | ✓    | ✓                |

### Instance Segmentation

| Model  | LoRA | SSF  | DoRA | Full Fine Tuning |
|--------|------|------|------|------------------|
| SegNeXT | ✓    | ✓    | ✓    | ✓                |

### Pose Detection

| Model  | LoRA | SSF  | DoRA | Full Fine Tuning |
|--------|------|------|------|------------------|
| RTMO   | ✓    | ✓    | ✓    | ✓                |

**Note that quantization support (QLoRA/QSSF) for adaptation of vision models is currently not supported.**
