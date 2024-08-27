# Nyuntam Text Generation

## Table of Contents

- [Nyuntam Text Generation](#nyuntam-text-generation)
   - [Overview](#overview)
   - [LLM Structured Pruning](#llm-structured-pruning)
      - [Fluctuation-based Adaptive Structured Pruning (FLAP)](#fluctuation-based-adaptive-structured-pruning-flap)
   - [LLM Quantization](#llm-quantization)
      - [W4A16 Activation aware Weight-Quantization (AWQ)](#w4a16-activation-aware-weight-quantization-awq)
      - [W4A8KV4 Quattuor-octo-Quattuor (QoQ)](#w4a8kv4-quattuor-octo-quattuor-qoq)
      - [W2A16 Additive Quantization of Language Models (AQLM)](#w2a16-additive-quantization-of-language-models-aqlm)
   - [LLM Engine](#llm-engine)
      - [TensorRT](#tensorrt)
         - [Model/Quantization Support Grid](#modelquantization-support-grid)
      - [ExLlama](#exllama)
      - [MLCLLM](#mlcllm)

## Overview

Nyuntam Text Generation is a comprehensive suite of tools and algorithms designed to optimize and accelerate the inference of large language models (LLMs) for text generation tasks. The suite encompasses various techniques such as structured pruning, quantization, and engine optimization, all aimed at enhancing the efficiency of LLMs. These tools are compatible with popular models like GPT-2, GPT-3, and BERT and can be deployed across a wide range of platforms, including CPUs, GPUs, and edge devices.

## LLM Structured Pruning

### Fluctuation-based Adaptive Structured Pruning (FLAP)

FLAP is an innovative framework that enhances the efficiency of Large Language Models (LLMs) by reducing storage requirements and improving inference speed. This framework outputs `model.safetensors`, which can be directly loaded using [`load_and_replace_weights`](https://github.com/nyunAI/nyuntam/blob/71f75e8c0b1b81a49758d3c2f87d89c99ff3124d/examples/text-generation/flap_pruning/convert_to_hf.py#L11).

**Parameters**

| Parameter                | Values   | Description   | Default Value  |    |       |
|------------------------- |--------- |------------------------------------------------------------------------- |--------------- |--- |--- |
| pruning_ratio            | Float    | Pruning ratio | 0.5            |    |       |
| metrics                  | "WIFV"   | Importance metric:<br>"WIFV" (Weighted Importance Feature Value)         | "WIFV"         |    |    |
| structure                | "AL-AM"  | Pruning structure:<br>"AL-AM" (Adaptive across both Layers and Modules)  | "AL-MM"        |    |    |
| remove_heads             | Int      | Number of heads to remove                                                | -1             |    |    |
| nsamples                 | Int      | Number of samples for evaluation                                         | 2048           |    |    |
| start_pruning_layer_idx  | Int      | Decoder Layer index to start pruning from.                               | 22             |    |    |

## LLM Quantization

### W4A16 Activation aware Weight-Quantization (AWQ)

LLM Quantization involves a 4-bit weight-only quantization method specifically designed for Language Model (LM) applications. This method uses GEMM (General Matrix Multiply) as the default operation. The process generates `*.safetensor` and `config.json` files that can be directly loaded by transformers' `AutoModelForCausalLM.from_pretrained()` or AutoAWQ's `AutoAWQForCausalLM.from_quantized()` for quantized models.

**Parameters**

| Parameter    | Values         | Description                                                                                                     | Default Value    |
|--------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| zero_point   | bool           | Whether to use [zero point](https://github.com/google/gemmlowp/blob/master/doc/quantization.md#domain-specific-constraint-the-real-value-0-must-be-exactly-representable). | True          |
| q_group_size | Int            | Quantization group size                                                                                         | 128              |
| w_bit        | Int            | Weight bitwidth (only 4-bit is supported)                                                                       | 4                |
| version      | "GEMM", "GEMV" | Version of AutoAWQ. One of GEMM or GEMV.                                                                        | "GEMM"           |

### W4A8KV4 Quattuor-octo-Quattuor (QoQ)

QoQ employs a 4-bit weight, 8-bit activation, and 4-bit KV cache configuration. The algorithm includes a progressive quantization strategy to reduce dequantization overhead and a SmoothAttention mechanism to mitigate accuracy loss from 4-bit KV quantization.

**Parameters**

| Parameter            | Values                                                                                 | Description                                                                                                                                                                                                                                        | Default Value     |    |    |
|--------------------- |--------------------------------------------------------------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |------------------ |--- |--- |
| save_model           | True                                                                                   | Whether to save the model                                                                                                                                                                                                                          | True              |    |    |
| keep_scales          | True                                                                                   | Whether to keep scales during quantization                                                                                                                                                                                                         | True              |    |    |
| loads_with_qserve    | False                                                                                  | Whether the model loads with [QServe](https://github.com/mit-han-lab/qserve)                                                                                                                                                                       | False             |    |    |
| quant_type           | "gchn",<br>"g128",<br>"awq",<br>"gptq",<br>"sq_dynamic",<br>"sq_static"  | Quantization type -<br>"gchn": QoQ Algorithm (Channelwise)<br>"g128": QoQ Algorithm (Groupwise)<br>"awq": AWQ Algorithm<br>"gptq": GPTQ Algorithm<br>"sq_dynamic": SmoothQuant Algorithm (Dynamic)<br>"sq_static": SmoothQuant Algorithm (Static)  | "gchn"            |    |    |
| eval.tasks           | arc_challenge:25                                                                       | Evaluation tasks                                                                                                                                                                                                                                   | arc_challenge:25  |    |    |
| eval.max_seq_length  | 4096                                                                                   | Maximum sequence length for evaluation                                                                                                                                                                                                             | 4096              |    |    |
| eval.evaluator       | "lm_eval"                                                                              | Evaluator used for evaluation                                                                                                                                                                                                                      | "lm_eval"         |    |    |

_Other nested config parameters can be updated as [here](https://github.com/nyunAI/nyuntam-text-generation/blob/9542023ef0836e4346c45eecaad83711c71efadc/scripts/quantisation/lmquant.yaml#L35). Find all the default configs for `quant_type` [here](https://github.com/nyunAI/nyuntam-text-generation/tree/9542023ef0836e4346c45eecaad83711c71efadc/quantization/mit_han_lab_lmquant/configs)._

### W2A16 Additive Quantization of Language Models (AQLM)

AQLM introduces learned additive quantization, tailored to each transformer block, and jointly optimizes codebook parameters across blocks. It stands out for being Pareto-optimal in accuracy vs. model size for models compressed to less than 3 bits per parameter. AQLM also offers practical, fast implementations for GPU and CPU, making it suitable for deploying LLMs on end-user devices.

**Parameters**

| Parameter                        | Values            | Description                                       | Default Value        |
|--------------------------------- |------------------ |-------------------------------------------------- |--------------------- |
| save_intermediate_results        | bool              | Whether to save intermediate results              | true                 |
| dtype                            | string            | Data type for quantization                        | "float16"            |
| Calibration Config               |                   |                                                   |                      |
| attn_implementation              | null or string    | Attention implementation                          | null                 |
| beam_size                        | int               | Beam size for calibration                         | 1                    |
| codebook_value_nbits             | int               | Number of bits for codebook values                | 16                   |
| codebook_value_num_groups        | int               | Number of groups for codebook values              | 1                    |
| dtype                            | string            | Data type for calibration                         | "float16"            |
| finetune_adam_beta1              | float             | Adam beta1 for finetuning                         | 0.9                  |
| finetune_adam_beta2              | float             | Adam beta2 for finetuning                         | 0.999                |
| finetune_batch_size              | int               | Batch size for finetuning                         | 16                   |
| finetune_early_stop              | int               | Early stopping criterion for finetuning           | 3                    |
| finetune_keep_best               | bool              | Whether to keep the best model during finetuning  | true                 |
| finetune_lr                      | float             | Learning rate for finetuning                      | 0.0001               |
| finetune_max_epochs              | int               | Maximum number of epochs for finetuning           | 25                   |
| in_group_size                    | int               | Input group size                                  | 8                    |
| init_max_iter                    | int               | Maximum iterations for initialization             | 100                  |
| init_max_points_per_centroid     | null or int       | Maximum points per centroid for initialization    | null                 |
| local_batch_size                 | int               | Local batch size                                  | 1                    |
| lr                               | float             | Learning rate                                     | 0.0001               |
| max_epochs                       | int               | Maximum number of epochs                          | 100                  |
| mix_compression                  | bool              | Whether to use mixed compression                  | false                |
| model_seqlen                     | int               | Model sequence length                             | 4096                 |
| nbits_per_codebook               | int               | Number of bits per codebook                       | 16                   |
| new_eval                         | bool              | Whether to use new evaluation                     | false                |
| no_quant                         | bool              | Whether to disable quantization                   | false                |
| nsamples                         | int               | Number of samples                                 | 2048                 |
| num_codebooks                    | int               | Number of codebooks                               | 1                    |
| offload_activations              | bool              | Whether to offload activations                    | true                 |
| on_save                          | null or function  | Function to call on save                          | null                 |
| out_group_size                   | int               | Output group size                                 | 1                    |
| print_frequency                  | int               | Frequency of printing                             | 10                   |
| relative_mse_tolerance           | float             | Relative MSE tolerance                            | 0.01                 |
| resume                           | bool              | Whether to resume training                        | false                |
| scale_nbits                      | int               | Number of bits for scaling                        | 0                    |
| seed                             | int               | Random seed                                       | 0                    |
| skip_out_loss                    | bool              | Whether to skip output loss                       | false                |
| steps_per_epoch                  | int               | Steps per epoch                                   | 100                  |
| true_sequential                  | bool              | Whether to use true sequential processing         | false                |
| trust_remote_code                | bool              | Whether to trust remote code                      | true                 |
| use_checkpointing                | bool              | Whether to use checkpointing                      | false                |
| use_faiss                        | bool              | Whether to use Faiss                              | false                |
| use_fast_tokenizer               | bool              | Whether to use fast tokenizer                     | false                |
| val_size                         | int               | Validation size                                   | 256                  |
| wandb                            | bool              | Whether to use Weights & Biases                   | false                |
| Finetune Config                  |                   |                                                   |                      |
| adam_beta1                       | float             | Adam beta1 for finetuning                         | 0.9                  |
| adam_beta2                       | float             | Adam beta2 for finetuning                         | 0.95                 |
| amp_dtype                        | string            | AMP data type                                     | float32              |
| amsgrad                          | bool              | Whether to use AMSGrad                            | false                |
| attn_implementation              | null or string    | Attention implementation for finetuning           | null                 |
| base_model                       | string            | Base model name                                   | base_model           |
| batch_size                       | int               | Batch size                                        | 1                    |
| beam_size                        | int               | Beam size                                         | 1                    |
| block_type                       | string            | Block type                                        | LlamaDecoderLayer    |
| code_adam_16bit                  | bool              | Whether to use 16-bit Adam for codes              | false                |
| code_beta1                       | float             | Beta1 for code optimization                       | 0.0                  |
| code_beta2                       | float             | Beta2 for code optimization                       | 0.95                 |
| code_dtype                       | string            | Data type for codes                               | uint16               |
| code_lr                          | float             | Learning rate for codes                           | 0.001                |
| code_selection_temperature       | float             | Temperature for code selection                    | 0                    |
| code_trust_ratio                 | float             | Trust ratio for codes                             | 0.01                 |
| debias                           | bool              | Whether to debias                                 | true                 |
| delta_decay                      | float             | Delta decay                                       | 0                    |
| download_num_workers             | null or int       | Number of workers for downloading                 | null                 |
| eval_datasets                    | list              | Evaluation datasets                               | ["wikitext2", "c4"]  |
| eval_every_steps                 | int               | Evaluate every n steps                            | 1                    |
| force_code_update                | bool              | Whether to force code update                      | false                |
| gradient_checkpointing           | bool              | Whether to use gradient checkpointing             | true                 |
| keep_best_model                  | bool              | Whether to keep the best model                    | false                |
| lamb                             | bool              | Whether to use LAMB optimizer                     | true                 |
| limit_parallel_inits             | int               | Limit on parallel initializations                 | 1                    |
| load_dtype                       | string            | Data type for loading                             | float32              |
| lr                               | float             | Learning rate                                     | 0.0001               |
| master_dtype                     | string            | Master data type                                  | float32              |
| max_code_change_per_step         | float             | Maximum code change per step                      | 0.01                 |
| max_epochs                       | int               | Maximum number of epochs                          | 10                   |
| microbatch_size                  | int               | Microbatch size                                   | 1                    |
| minimize_sync                    | bool              | Whether to minimize synchronization               | false                |
| model_seqlen                     | int               | Model sequence length                             | 4096                 |
| monkeypatch_old_pickle           | bool              | Whether to monkeypatch old pickle                 | false                |
| num_workers                      | int               | Number of workers                                 | 8                    |
| overwrite_cache                  | bool              | Whether to overwrite cache                        | false                |
| preprocessing_chunk_length       | null or int       | Preprocessing chunk length                        | null                 |
| preprocessing_keep_in_memory     | bool              | Whether to keep preprocessing in memory           | false                |
| preprocessing_num_workers        | int               | Number of preprocessing workers                   | 24                   |
| print_every_steps                | int               | Print every n steps                               | 1                    |
| save_every_steps                 | int               | Save every n steps                                | 1                    |
| seed                             | int               | Random seed                                       | 1337                 |
| straight_through_buffer_dtype    | string            | Straight-through buffer data type                 | float32              |
| trust_remote_code                | bool              | Whether to trust remote code                      | true                 |
| update_codebooks_and_scales      | bool              | Whether to update codebooks and scales            | true                 |
| update_codes                     | bool              | Whether to update codes                           | true                 |
| update_non_quantized_parameters  | bool              | Whether to update non-quantized parameters        | true                 |
| use_fast_tokenizer               | bool              | Whether to use fast tokenizer                     | false                |
| use_fsdp_amp                     | bool              | Whether to use FSDP AMP                           | false                |
| verbose_optimizer                | bool              | Whether to use verbose optimizer                  | true                 |
| wandb                            | bool              | Whether to use Weights & Biases                   | false                |
| wrap_separately                  | list              | Layers to wrap separately                         | []                   |
| Conversion Config                |                   |                                                   |                      |
| attn_implementation              | null or string    | Attention implementation for conversion           | null                 |
| code_dtype                       | string            | Data type for codes                               | int32                |
| load_dtype                       | string            | Data type for loading                             | auto                 |
| trust_remote_code                | bool              | Whether to trust remote code for conversion       | true                 |

## LLM Engine

### TensorRT

The LLM Engine TensorRT optimizes LLMs for inference by building TensorRT engines equipped with state-of-the-art optimizations to ensure efficient inference performance. The output is a `.engine` model file that can be directly loaded by NVIDIA Triton Inference Server.

**Parameters**

| Parameter    | Values                                   | Description                                                                                 | Default Value    |
|--------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| to_quantize  | bool                                     | Whether to quantize the model before building the engine.                                   | True             |
| quant_method | "fp8", "int4_awq", "smoothquant", "int8" | Quantization format                                                                         | "int4_awq"       |
| smoothquant  | float                                    | (if quant_method = "smoothquant") The smooth quant's α value, which controls quantization difficulty migration between activations and weights.        | 0.5           |
| calib_size   | Int                                      | Calibration size                                                                            | 32               |
| dtype        | "float16"                                | The data type of the model                                                                  | "float16"        |

#### Model/Quantization Support Grid

| Model      | fp8 | int4_awq | smoothquant | int8 |
|------------|-----|----------|-------------|------|
| LLaMA      | ✓   | ✓        | ✓           | ✓    |
| LLaMA-2    | ✓   | ✓        | ✓           | ✓    |
| Vicuna     | ✓   | ✓        | ✓           | ✓    |
| Mixtral    | ✓   | ✓        | -           | ✓    |
| Mistral-7B | ✓   | ✓        | -           | ✓    |
| Gemma      | ✓   | ✓        | -           | ✓    |

### ExLlama

LLM Engine ExLlama introduces a new quantization format known as EXL2, providing flexibility in weight storage. This implementation generates engine files and a script for fast inference on the given model. The output includes `.safetensor`, `config.json` model files, and a `run.sh` script for test inference using ExllamaV2.

**Parameters**

| Parameter            | Values                 | Description                                                | Default Value                                                 |
|----------------------|------------------------|------------------------------------------------------------|---------------------------------------------------------------|
| bits                 | Float >= 2 , <= 8      | Target bits per weight                                      | 4.125                                                         |
| shard_size           | Int                    | Maximum shard size in MB while saving the model             | 8192  |
| rope_scale           | Float                  | RoPE scaling factor (related to RoPE (NTK) parameters)      | 1     |
| rope_alpha           | Float                  | RoPE alpha value (related to RoPE (NTK) parameters)         | 1     |
| head_bits            | Int                    | Target bits per weight (for the head layer)                 | 6     |

### MLCLLM

The LLM Engine MLCLLM offers compiler accelerations and runtime optimizations for native deployment across various platforms, including edge devices. The output consists of `params-*.bin` files and compiled files that can be directly used by MLC Chat, along with a `run.py` script for sample usage.

**Parameters**

| Parameter       | Values                   | Description                                                | Default Value        |
|-----------------|--------------------------|------------------------------------------------------------|----------------------|
| quantize        | bool                      | Indicates whether quantization is applied to the model      | True                 |
| quant_method    | "q4f16_0", "q4f16_autoawq"| Method used for quantization                                | "q4f16_autoawq"      |
| conv_template   | "llama-2"                 | [Conversation templates](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_chat/conversation_template.py) | None                 |
| llvm_triple     | null                      | LLVM triple                                                 | None                 |
