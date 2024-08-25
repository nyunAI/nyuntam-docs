# Model Compression & Adaption Support Grid

Below are tables summarizing the support for various compression and adaption techniques across different models. Apart from these, other similar models may be supported but have not been tested. 

## Vision Compression

Note that in the Table below, CPU and GPU indicate the target device of deployment. PTQ indicates Post Training Quantization and QAT indicates Quantization Aware Training.

| Model              | CPU PTQ - Torch | CPU PTQ - OpenVino | CPU PTQ - ONNX | GPU PTQ - TensorRT | CPU QAT - Torch | Knowledge Distillation | Structured Pruning | CPU QAT - OpenVino |
|------------------------------|-----------------|--------------------|----------------|--------------------|-----------------|------------------------|--------------------|--------------------|
| Resnet (timm)                | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| Convnextv2 (huggingface)     | -               | ✓                  | ✓              | ✓                  | -               | ✓                      | ✓                  | ✓                  |
| Mobilenetv3 (timm)           | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| DeiT (huggingface)           | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| VanillaNet (timm)            | -               | -                  | ✓              | ✓                  | -               | ✓                      | ✓                  | ✓                  |
| Swin (huggingface)           | ✓               | ✓                  | ✓              | ✓                  | ✓               | ✓                      | ✓                  | ✓                  |
| YoloX (mmyolo/mmdet)         | -               | ✓                  | -              | ✓                  | ✓               | ✓                      | ✓                  | -                  |
| RTMDet (mmyolo/mmdet)        | -               | ✓                  | -              | ✓                  | ✓               | ✓                      | -                  | -                  |
| Yolov8 (mmyolo)              | -               | -                  | -              | ✓                  | -               | ✓                      | -                  | -                  |

## LLM Compression

| Model   | LLM Quantization | LLM Engine TensorRT | LLM Engine Exllama | LLM Engine MLC-LLM | LLM Structured Pruning |
|---------|------------------|---------------------|--------------------|--------------------|------------------------|
| LLaMA   | ✓                | ✓                   | ✓                  | ✓                  | ✓                      |
| LlaMA-2 | ✓                | ✓                   | ✓                  | ✓                  | ✓                      |
| Vicuna  | ✓                | ✓                   | ✓                  | ✓                  | ✓                      |
| Mistral | ✓                | ✓                   | ✓                  | ✓                  | -                      |
| Mixtral | ✓                | ✓                   | ✓                  | ✓                  | -                      |
| Gemma   | ✓                | ✓                   | -                  | -                  | -                      |

## Adapt

### LLM Tasks 
<ul>
<li> Text Generation </li>
<li> Summarization </li>
<li> Question Answering </li>
<li> Text Classification </li>
<li>Translation </li>
</ul> 

**All of the major huggingface models are supported for these tasks.**

### Image Classification

**All major image models on huggingface and timm are supported in Adapt.**

### Object Detection
| | LoRA | SSF | DoRA | Full Fine Tuning |
|---------|------------------|---------------------|--------------------|--------------------| 
| YoloX | ✓ | ✓ | ✓ | ✓ |
| RTMDet | ✓ | ✓ | ✓ | ✓ |

### Instance Segmentation
| | LoRA | SSF | DoRA | Full Fine Tuning |
|---------|------------------|---------------------|--------------------|--------------------| 
| SegNeXT | ✓ | ✓ | ✓ | ✓ |

### Pose Detection
| | LoRA | SSF | DoRA | Full Fine Tuning |
|---------|------------------|---------------------|--------------------|--------------------| 
| RTMO | ✓ | ✓ | ✓ | ✓ |

**Note that quantization support (QLoRA/QSSF) for adaptation of vision models is currently not supported.**

