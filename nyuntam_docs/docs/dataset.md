# Dataset Importation

Nyuntam provides comprehensive support for various custom dataset formats. Additionally, some algorithms can execute without requiring any custom datasets.

## Dataset Preparation Guidelines

### Nyuntam Text-Generation

Nyuntam supports loading any text dataset compatible with the Hugging Face `datasets.load_dataset` for HF datasets or `datasets.load_from_disk` for custom datasets. It supports two main dataset formats:

**LLM - Single Column**

This format is suitable for use cases where the dataset is already formatted with a single text column. For example, "wikitext" dataset with "text" as the `TEXT_COLUMN`.

**LLM - Multi Columns**

This format is suitable for datasets with multiple columns. The multi-column dataset can also handle simple formatting of the dataset for instructional use cases. (See example usage below)

Please note that there are no limitations regarding the loading of datasets with either single-column or multi-column structures. However, it is advisable to opt for loading multi-column datasets using option 2, especially when straightforward formatting is desired.

**Parameters for Dataset Loading:**

| Parameter         | Default Value | Description                                                                  |
|-------------------|---------------|------------------------------------------------------------------------------|
| DATASET_SUBNAME   | null          | Subname of the dataset if applicable.                                        |
| TEXT_COLUMN       | text          | Specifies the text columns to be used.<br>If multiple columns are present, they should be<br>separated by commas. |
| SPLIT             | train         | Specifies the split of the dataset to load,<br>such as 'train', 'validation', or 'test'. |
| FORMAT_STRING     | null          | If provided, this string is<br>used to format the dataset.<br>It allows for customization<br>of the dataset's text representation. |

**Dataset Formatting:**

The process responsible for formatting the dataset based on the provided parameters follows these steps:

1. If no format string is provided, the `TEXT_COLUMN` and the dataset are used as is.
2. If a format string is provided, it is applied to format the dataset. The format string should contain placeholders for the columns specified in `TEXT_COLUMN`.
3. Once the dataset is formatted, a log message is generated to indicate that the format string was found and used. It also displays a sample of the formatted dataset.
4. Finally, the dataset is mapped to replace the original "text" columns with the newly formatted "text" column (or) a new "text" column is created.

**Example usage for Multi-column dataset:**

Params for Alpaca Dataset (`yahma/alpaca-cleaned`)
```shell
DATASET_SUBNAME - null
TEXT_COLUMN - input,output,instruction
SPLIT - train
FORMAT_STRING - Instruction:\n{instruction}\n\nInput:\n{input}\n\nOutput:\n{output}
```

Alpaca dataset before Formatting

| input                           | instruction                     | output                          |
|---------------------------------|---------------------------------|---------------------------------|
| How to bake a cake              | Step 1: Preheat the oven to...  | A delicious homemade cake...    |
| Introduction to Python          | Python is a high-level...       | Learn Python programming...     |
| History of the Roman Empire     | The Roman Empire was...         | Explore the rise and fall...    |
| ...                             | ...                             | ...                             |

Alpaca dataset after formatting with the input params

| input                           | instruction                     | output                          | text                            |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| How to bake a cake              | Step 1: Preheat the oven to...  | A delicious homemade cake...    | Instruction:\nStep 1: Preheat...\n\nInput:\nHow to bake a cake\n\nOutput:\nA delicious homemade cake...  |
| Introduction to Python          | Python is a high-level...       | Learn Python programming...     | Instruction:\nPython is a hi...\n\nInput:\nIntroduction to Python\n\nOutput:\nLearn Python programming...  |
| History of the Roman Empire     | The Roman Empire was...         | Explore the rise and fall...    | Instruction:\nThe Roman Emp...\n\nInput:\nHistory of the Roman Empire\n\nOutput:\nExplore the rise and fall...  |
| ...                             | ...                             | ...                             | ...                             |

## Importing Your Dataset

There are two different ways to import your dataset into Nyuntam:

### Custom Data
Users who are using a custom dataset (stored locally):
- For nyuntam-adapt, `CUSTOM_DATASET_PATH` argument in the yaml can be updated . 

- For nyuntam-text-generation and nyuntam-vision, `DATA_PATH` argument in the yaml can be updated.

### Pre-existing dataset

Users who are using existing datasets from huggingface:
- For nyuntam-adapt, use the `DATASET` argument in the yaml. 
- for nyuntam-text-generation, use the `DATASET_NAME` argument in the yaml.

For more examples please refer to [Examples](./examples/index.md) 


**Note:** For LLM tasks, the data folder must be loadable by `datasets.load_from_disk` and should return a `datasets.DatasetDict` object and not a `datasets.Dataset` object.