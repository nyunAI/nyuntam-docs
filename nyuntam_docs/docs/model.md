# Model Importation

Nyuntam facilitates the seamless import of pre-trained or fine-tuned models for compression and adaptation.

## Preparing Your Model

Users can directly use a pre-trained or fine-tuned model for compression and adaptation.

## Importing Your Custom Weights

Nyuntam offers two methods for importing your model:

### Using Models Hosted on the Internet

To use this method:

1. Ensure that the model is present on Hugging Face (for LLM tasks) or OpenMMLab's repository (for vision tasks).
2. Use the `MODEL_PATH` argument in the YAML to specify the model path. See [here](./examples/adapt/text_generation/config.yaml) for examples.

### Using Model Weights Stored Locally

To use this method:

1.For nyuntam-adapt use the `LOCAL_MODEL_PATH` argument in the YAML to specify the absolute path of the folder containing the model weights.
2. For nyuntam-text-generation and nyuntam-vison use the `CUSTOM_MODEL_PATH` argument in the YAML to specify the absolute path of the folder containing the model weights.
3. When using distillation algorithms with nyuntam-vision, custom model weights can be loaded using `CUSTOM_TEACHER_PATH`. 


**Note:** For LLM tasks, the model folder must be loadable by `transformers.AutoModelForCausalLM.from_pretrained` and should return a `torch.nn.Module` object or a state_dict (i.e., `collections.OrderedDict` object) that can be loaded using `torch.load`.
