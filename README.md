# resnet50-alexnet-egg-classification

> A CNN-based solution to classify eggs as damaged or not damaged, using the ResNet50 and AlexNet models with PyTorch.

## About the Project

This project implements a solution based on Convolutional Neural Networks (CNNs) to classify images of eggs into two categories: `damaged` and `not_damaged`. The training and evaluation are performed using two well-known models, ResNet50 and AlexNet, implemented with the PyTorch library.

This work is also documented in the following resources:
*   **[Project Article (Portuguese)](https://github.com/luizvilasboas/resnet50-alexnet-egg-classification/blob/main/docs/article.pdf)**
*   **[Explanatory Video (Portuguese)](https://www.youtube.com/watch?v=Ijp6jcghPM8)**

## Tech Stack

*   [Python](https://www.python.org/)
*   [PyTorch](https://pytorch.org/)
*   [CUDA](https://developer.nvidia.com/cuda-toolkit) (Optional, for GPU acceleration)

## Usage

Below are the instructions for you to set up and run the project locally.

### Prerequisites

You need to have the following software installed:

*   [Python](https://www.python.org/downloads/) (3.8 or higher)
*   [PyTorch](https://pytorch.org/get-started/locally/) (2.0 or higher)

### Installation and Setup

Follow the steps below:

1.  **Clone the repository**
    ```bash
    git clone https://github.com/luizvilasboas/resnet50-alexnet-egg-classification.git
    ```

2.  **Navigate to the project directory**
    ```bash
    cd resnet50-alexnet-egg-classification
    ```

3.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    *(On Windows, use `venv\Scripts\activate`)*

4.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the dataset**
    The dataset for this project can be found on Kaggle: [Eggs Images Classification (Damaged or Not)](https://www.kaggle.com/datasets/abdullahkhanuet22/eggs-images-classification-damaged-or-not). Download it and ensure the `dataset/` directory is configured correctly.

### Workflow

To train the models, run the corresponding training scripts:

*   **Train AlexNet:**
    ```bash
    python3 train_alexnet.py
    ```
*   **Train ResNet50:**
    ```bash
    python3 train_resnet50.py
    ```
Training results, including accuracy and loss graphs, will be saved automatically in the `output/` directory.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
