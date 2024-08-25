# Nyuntam

## Overview

Nyuntam is an efficiency-focused deep learning platform designed to streamline the entire development-to-deployment cycle of deep learning models. Zero offers multiple plugins covering common use cases in deep learning production cycles:

1. **[nyuntam_adapt](./nyuntam_adapt/index.md)**: A robust adaptation module that effortlessly fine-tunes and performs transfer learning on a diverse range of deep learning tasks and models. With nyuntam-adapt, users can adapt large and medium-sized vision and language models seamlessly, enabling the creation of custom models with just a few clicks. This module incorporates state-of-the-art adaptation methods such as (Q)-LoRA, SSF, etc., allowing users to strike the optimal balance across various model metrics, including parameter count and training throughput.

2. **[nyuntam_vision](./nyuntam_vision/index.md)**: Developed to compress and optimize computer vision models , nyuntam_vision provides a set of compression techniques tailored for specific deployment constraints. Users have the flexibility to choose and combine multiple techniques to achieve the best trade-off between model performance and deployment constraints. Leveraging cutting-edge techniques like pruning, quantization, distillation, etc., nyuntam_vision achieves exceptional model compression levels on a variety of vision models across various frameworks.

3. **[nyuntam_text_generation](./nyuntam_text_generation/index.md)**:  Developed to compress and optimize Large Language Models , nyuntam_text_generation provides a set of compression techniques tailored for specific deployment constraints. Users have the flexibility to choose and combine multiple techniques to achieve the best trade-off between model performance and deployment constraints. Leveraging cutting-edge techniques like pruning, quantization, distillation, etc., nyuntam_text_generation achieves exceptional model compression levels on a variety of LLMs across various frameworks.


## Setup and Installation

For hassfle-free experimentation and quick results, nyuntam provides a Command Line Interface tool : [nyunzero-cli](https://github.com/nyunAI/nyunzero-cli). The doumentation about using the cli can be found [here](./nyunzero_cli.md)

Nyuntam is a fully opensource project and users can go through the code and contribute at [nyunam](https://github.com/nyunAI/nyuntam).

## About NyunAI

NyunAI began its journey in 2020 with state-of-the-art research on model compression. Recognizing the challenges in setting up pipelines for model compression and downstream task adaptation, NyunAI aimed to simplify the process for researchers and developers, allowing them to efficiently build and deploy models while focusing entirely on problem-solving. Since then, NyunAI has remained committed to developing efficient deep learning technology and supporting software.

## Contact NyunAI

NyunAI's support team is always available to assist users with any questions or concerns. Users can reach out to the team via email at [support@nyunai.com](mailto:support@nyunai.com).