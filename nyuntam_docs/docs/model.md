# Model Importation

Nyuntam facilitates the seamless importation of pre-trained or fine-tuned models for compression and adaptation.

## Preparing Your Model

Users can directly utilize a pre-trained or fine-tuned model for compression and adaptation.

## Importing Your Custom Weights

Nyuntam offers two methods for importing your model:

### Using Models hosted on the internet 

To employ this method:

1. Ensure that the model is present on huggingface (for LLM tasks) or on OpenMMLab's repository (for vision task). 
2. Use the `MODEL_PATH` argument in the yaml to specify the model path. See [here](./examples/adapt/text_generation/config.yaml) for examples. 

### Using Model weights stored locally

To employ this method:

1. Use the `LOCAL_MODEL_PATH` argument in the yaml to specify the absolute path of the folder containing the model weights.


**Note:** For LLM tasks, the model folder must be loadable by `transformers.AutoModelForCausalLM.from_pretrained` and should return a `torch.nn.Module` object or a state_dict (i.e., `collections.OrderedDict` object) that can be loaded using `torch.load`.