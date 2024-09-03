# Dataset Importation

Nyuntam provides comprehensive support for various custom dataset formats. Additionally, some algorithms can execute without requiring any custom datasets.

## Dataset Preparation Guidelines

### Image Classification
For image classification tasks, nyuntam is compatible with the standard ImageNet dataset format. This format necessitates organizing images into folders representing respective categories, with validation splits already performed. The prescribed structure is as follows:

```shell
ImageNetDataset/
| -- train
|   | -- Class_A
|   |   | -- image1.jpg
|   |   | -- image2.jpg
|   |   | -- ...
|   | -- Class_B
|   |   | -- image1.jpg
|   |   | -- image2.jpg
|   |   | -- ...
|   | -- ...
| -- val
|   | -- Class_A
|   |   | -- image1.jpg
|   |   | -- image2.jpg
|   |   | -- ...
|   | -- Class_B
|   |   | -- image1.jpg
|   |   | -- image2.jpg
|   |   | -- ...
|   | -- ...
```

### Object Detection
Nyuntam extends support to both COCO and VOC dataset formats for object detection tasks. The formats are detailed below:

**COCO Format**
```shell
CocoDataset/
| -- train2017
|   | -- (Contains training images)
| -- val2017
|   | -- (Contains validation images)
| -- annotations
|   | -- instances_train2017.json  (Train Annotations)
|   | -- instances_val2017.json    (Val Annotations)
```

**VOC Format**
```shell
VOCdevkit/
| -- VOC2012
|   | -- Annotations
|   |   | -- (Contains annotation files)
|   | -- JPEGImages
|   |   | -- (Contains JPEG images)
```

### Segmentation
For image segmentation tasks, Nyuntam Vision exclusively supports the VOC Format. The layout is structured as follows:

```shell
VOCdevkit/
| -- VOC2012
|   | -- Annotations
|   |   | -- (Contains annotation files)
|   | -- JPEGImages
|   |   | -- (Contains JPEG images)
|   | -- ImageSets
|   |   | -- (Contains ImageSets files)
|   | -- SegmentationClass
|   |   | -- (Contains SegmentationClass files)
|   | -- SegmentationObject
|   |   | -- (Contains SegmentationObject files)
```

Nyuntam Adapt requires the dataset to be in the following format : 
```shell
custom_data/
| -- dataset
|   | -- images (images in jpg format)
|   | -- labels (segmentation maps in png format)
|   | -- splits 
|   |   | -- train.txt (names of the images for training (without the extension))
|   |   | -- val.txt (names of the images for validation (without the extension))
```

### Pose Detection
For pose detection tasks, Nyuntam exclusively supports the COCO-Pose Format. The layout is structured as follows:

```shell
coco-pose/
| -- images
|   | -- train2017
|   |   | -- Contains training images(JPEG format)
|   | -- val2017
|   |   | -- Contains testing images (JPEG images)
| -- annotations
|   |   | -- person_keypoints_train2017.json (Contains annotation file in coco-pose format)
|   |   | -- person_keypoints_val2017.json (Contains annotation file in coco-pose format)
```


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

### Text Classification
Nyuntam supports token classification and text clasification. Users can load any text dataset compatible with the Hugging Face `datasets.load_dataset` for HF datasets or `datasets.load_from_disk` for custom datasets. Make sure the uploaded dataset has the following format:


| Parameter        | Data Type  | Default Value            | Description                                      |
|------------------|------------|--------------------------|--------------------------------------------------|
| input_column          | str     | "text"                   | The name of the input text column : **"tokens"** - for token classification (new, pos, chunk) and **"text"** - for text classification|
| target_column          | str     | "label"                    | the target column in the dataset : **"ner_tags"** : NER , **"pos_tags"**: POS tagging, **"chunk_tags"**: Chunking, **"label"** : text classification|

Example dataset formats : 

**TEXT CLASSIFICATION DATASET**

| text | label |
|------|-------|
| I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it when it was first released in 1967. I also... | 0  |
| "I Am Curious: Yellow" is a risible and pretentious steaming pile. It doesn't matter what one's political views are because this film can hardly be taken... | 0  |
| If only to avoid making this type of film in the future. This film is interesting as an experiment but tells no cogent story.<br /><br />One might... | 0  |
| This film was probably inspired by Godard's Masculin, féminin and I urge you to see that film instead.<br /><br />The film has two strong elements and... | 0  |
| Oh, brother...after hearing about this ridiculous film for umpteen years all I can think of is that old Peggy Lee song..<br /><br />"Is that all there is?"... | 0  |

**TOKEN CLASSIFICATION DATASET**

| id | tokens | pos_tags |
|----|--------|----------|
| 0 | ["EU", "rejects", "German", "call", "to", ...] | [22, 42, 16, 21, 35, 37, 16, 21, ...] |
| 1 | ["Peter", "Blackburn"] | [22, 22] |
| 2 | ["BRUSSELS", "1996-08-22"] | [22, 11] |
| 3 | ["The", "European", "Commission", "said", "on", ...] | [12, 22, 22, 38, 15, 22, 28, 38, ...] |
| 4 | ["Germany", "'s", "representative", "to", ...] | [22, 27, 21, 35, 12, 22, 22, 27, ...] |

### Summarization
Nyuntam supports loading any text dataset compatible with the Hugging Face `datasets.load_dataset` for HF datasets or `datasets.load_from_disk` for custom datasets. Make sure the uploaded dataset has the following format:

| document | summary |
|----------|---------|
| The full cost of damage in Newton Stewart, one of the areas worst affected, is still... | Clean-up operations are continuing across the Scottish Borders and Dumfries and... |
| A fire alarm went off at the Holiday Inn in Hope Street at about 04:20 BST on Saturday... | Two tourist buses have been destroyed by fire in a suspected arson attack in Belfas... |
| Ferrari appeared in a position to challenge until the final laps, when the Mercedes... | Lewis Hamilton stormed to pole position at the Bahrain Grand Prix ahead of Mercedes... |
| John Edward Bates, formerly of Spalding, Lincolnshire, but now living in London,... | A former Lincolnshire Police officer carried out a series of sex attacks on... |
| Patients and staff were evacuated from Cerahpasa hospital on Wednesday after a man... | An armed man who locked himself into a room at a psychiatric hospital in Istanbul has... |




The dataset params for summarization tasks in nyuntam-adapt are as follows : 

| Key                   | Value                          | Description                                   |
|-----------------------|--------------------------------|-----------------------------------------------|
| DATASET_SUBNAME   | null          | Subname of the dataset if applicable.                                        |
| input_column          | 'document'                     | Name of the input column containing the text corpus                     |
| target_column         | 'summary'                      | Name of the target column containing the summarized text                    |

### Question Answering
The dataset for question answering must follow the general nyuntam-adapt dataset. nyuntam-adapt currently support extractive question-answering and hence requires :

- **CONTEXT** 
        - Text column that contains the context 
- **QUESTION**
        - Contains the Question 
- **ANSWER**
        - A column containing dictionary entries with the answer and index of context from which the answer starts. <br>
        example : 
        ```
        { "text": [ "...answer text..." ], "answer_start": [ 276 ] }
        ```

The column names can be different from this, but they should be mentioned in 

  - **input_column** : 'context'
  - **input_question_column** : 'question'
  - **target_column** : 'answer'

Dataset arguments in question answering: 

| Key                   | Value                          | Description                                   |
|-----------------------|--------------------------------|-----------------------------------------------|
| DATASET_SUBNAME   | null          | Subname of the dataset if applicable.                                        |
| input_column          | 'context'                      | Name of the input column                     |
| input_question_column | 'question'                     | Name of the input question column            |
| target_column         | 'answer'                       | Name of the target column                    |
| squad_v2_format       | False #True                    | Whether the data follows SQUAD-V2 Format (True/False)    |

Example of a default dataset for Question Answering :

| context |question | answers |
|---------|-----------|---------|
|Beyoncé Giselle Knowles-Carter ...|When did Beyonce start becoming... |{ "text": [ "in the late 1990s" ],...} |
|Beyoncé Giselle Knowles-Carter |What areas did Beyonce compete in... |{ "text": [ "singing and dancing" ],... }|
| Beyoncé Giselle Knowles-Carter | When did Beyonce leave Destiny's Chil... |{ "text": [ "2003" ], "answer_start": [ 526 ...] }|
 Beyoncé Giselle Knowles-Carter | In what city and state did Beyonce... |{ "text": [ "Houston, Texas" ],... }|

### Translation
Nyuntam supports loading any text dataset compatible with the Hugging Face `datasets.load_dataset` for HF datasets or `datasets.load_from_disk` for custom datasets. Make sure the uploaded dataset has the following format:

| id | translation |
|----|-----------------------------------------|
| 0 | { "ca": "Source: Project GutenbergTranslation: Josep Carner", "de": "Source: Project Gutenberg" } |
| 1 | { "ca": "Les Aventures De Tom Sawyer", "de": "Die Abenteuer Tom Sawyers" } |
| 2 | { "ca": "Mark Twain", "de": "Mark Twain" } |
| 3 | { "ca": "PREFACI.", "de": "Vorwort des Autors." } |
| 4 | { "ca": "La major part de les aventures consignades en aquest llibre succeïren de bo de bo; una o dues són experiments de la meva collita; la resta pertanye...", "de": "" } |

Dataset Arguments in Translation:

 Key                   | Value                          | Description                                   |
|-----------------------|--------------------------------|-----------------------------------------------|
| source_lang     | ''                           |The key of source language as per the given dataset           |
| target_lang      | ''                           |The key of target language as per the given dataset            |
| DATASET_SUBNAME   | null          | Subset for the dataset (for multilingual datasets, the subname generally represents the pair of language used for translation)                                        |

## Importing Your Dataset

There are two different ways to import your dataset into Nyuntam:

### Custom Data

Users who are using a custom dataset (stored locally) to finetune a model can use the `CUSTOM_DATASET_PATH` argument in the yaml to do so. 

### Pre-existing dataset

Users who are using existing datasets from huggingface can use the `DATASET` argument in the yaml. 

For more examples please refer to [Examples](./examples/index.md) 


**Note:** For LLM tasks, the data folder must be loadable by `datasets.load_from_disk` and should return a `datasets.DatasetDict` object and not a `datasets.Dataset` object.