### nyuntam_text_generation

#### LLM Structured Pruning
LLM Structured Pruning is a novel structured pruning framework for Large Language Models (LLMs) that improves efficiency by reducing storage and enhancing inference speed. Outputs `model.safetensors`, directly loadable by `transformers.from_pretrained()`. 

| Parameter     | Values                                      | Description                                                                                                                                                                                                                                                | Default Value |
|---------------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| pruning_ratio | Float                                       | Pruning ratio                                                                                                                                                                                                                                     | 0.2           |
| metrics       | "IFV",<br>"WIFV",<br>"WIFN"                 | Importance metric: <br>"WIFN" (Weighted Importance Feature Norm), <br>"IFV" (Importance Feature Value), <br>"WIFV" (Weighted Importance Feature Value)                                                                                                     | "WIFV"        |
| structure     | "UL-UM",<br>"UL-MM",<br>"AL-MM",<br>"AL-AM" | Pruning structure:<br>"UL-UM" (Uniform across Layers, Uniform across Modules),<br>"UL-MM" (Uniform across Layers, Manual ratio for Modules),<br>"AL-MM" (Adaptive across Layers, Manual for Modules),<br>"AL-AM" (Adaptive across both Layers and Modules) | "AL-MM"       |
| remove_heads  | Int                                         | Number of heads to remove                                                                                                                                                                                                                                  | 8             |
| nsamples      | Int                                         | Number of samples for evaluation                                                                                                                                                                                                                           | 2048          |


#### LLM Quantization
A 4-bit weight-only quantization method designed for Language Model (LM) applications. Utilizes GEMM (General Matrix Multiply) as the default operation. Generates `*.safetensor` & `config.json` files that can be directly loaded by transformers' `AutoModelForCausalLM.from_pretrained()` or AutoAWQ's `AutoAWQForCausalLM.from_quantized()` for quantized models.


| Parameter    | Values         | Description                                                                                                                                                                | Default Value |
|--------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| zero_point   | bool           | Whether to use [zero point](https://github.com/google/gemmlowp/blob/master/doc/quantization.md#domain-specific-constraint-the-real-value-0-must-be-exactly-representable). | True          |
| q_group_size | Int            | Quantization group size                                                                                                                                                    | 128           |
| w_bit        | Int            | Weight bitwidth (only 4 bit is supported)                                                                                                                                  | 4             |
| version      | "GEMM", "GEMV" | Version of AutoAWQ. One of GEMM or GEMV.                                                                                                                                   | "GEMM"        |


#### LLM Engine TensorRT
Optimizes LLMs for inference and builds TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently. Outputs `.engine` model files which can be directly loaded by NVIDIA Triton Inference Server.

| Parameter    | Values                                   | Description                                                                                                                                            | Default Value |
|--------------|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| to_quantize  | bool                                     | To first quantize the model<br>and then build engine.                                                                                                  | True          |
| quant_method | "fp8", "int4_awq", "smoothquant", "int8" | Quantization format                                                                                                                                    | "int4_awq"    |
| smoothquant  | float                                    | (if quant_method = "smoothquant")<br>smooth quant's α value <br>(to control quantization <br>difficulty migration between <br>activations and weights) | 0.5           |
| calib_size   | Int                                      | Calibration size                                                                                                                                       | 32            |
| dtype        | "float16"                                | dtype of the model                                                                                                                                     | "float16"     |

1. Model/Quantization Support Grid:

| Model      | fp8 | int4_awq | smoothquant | int8 |
|------------|-----|----------|-------------|------|
| LLaMA      | ✓   | ✓        | ✓           | ✓    |
| LLaMA-2    | ✓   | ✓        | ✓           | ✓    |
| Vicuna     | ✓   | ✓        | ✓           | ✓    |
| Mixtral    | ✓   | ✓        | -           | ✓    |
| Mistral-7B | ✓   | ✓        | -           | ✓    |
| Gemma      | ✓   | ✓        | -           | ✓    |

#### LLM Engine ExLlama
A new quantization format introducing EXL2, which brings a lot of flexibility to how weights are stored. This implementation generates the engine files and a script required to produce fast inferences on the provided model. Outputs `.safetonsor`, `config.json` model files along with `run.sh` that loads and runs a test inference with ExllamaV2.

| Parameter            | Values                 | Description                                                | Default Value                                                 |
|----------------------|------------------------|------------------------------------------------------------|---------------------------------------------------------------|
| bits                  | Float >= 2 , <= 8      | Target bits per weight                                    | 4.125                                                         |
| shard_size            | Int                    | Max shard size in MB while saving model                   | 8192                                                          |
| rope_scale            | Float                  | RoPE scaling factor (related to RoPE (NTK) parameters for calibration)| 1                                                 |
| rope_alpha            | Float                  | RoPE alpha value (related to RoPE (NTK) parameters for calibration)   | 1                                                 |
| head_bits             | Int                    | Target bits per weight (for head layer)                   | 6                                                             |

#### LLM Engine MLCLLM
Compiler accelerations and runtime optimizations for native deployment across platforms and edge devices. Outputs `params-*.bin` files and compiled files directly usable by MLC Chat. Also produces a `run.py` for sample usage.

| Parameter       | Values                   | Description                                               | Default Value        |
|------------------|---------------------------|-----------------------------------------------------------|-----------------------|
| quantize         | bool                     | Indicates whether quantization is applied to the model    | True                  |
| quant_method     | "q4f16_0", "q4f16_autoawq" | Method used for quantization | "q4f16_autoawq"            |
| conv_template    | "llama-2"                | [Conversation templates](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_chat/conversation_template.py)| None                  |
| llvm_triple      | null                     | LLVM triple                                              | None                  |