# COCO Classification with CLIP

This project implements a **Few-Shot Classification** model using OpenAI's **CLIP** (Contrastive Language-Image Pre-Training) on a subset of the **COCO** dataset. It leverages a pre-trained CLIP backbone (`ViT-L/14@336px`) as a powerful feature extractor and trains a lightweight linear classifier head to categorize images into 10 target classes.

## Features

- **CLIP Integration**: Uses OpenAI's state-of-the-art CLIP model for robust image feature extraction.
- **Few-Shot Learning**: Configured to work effectively with limited data samples per class.
- **Customizable**: Easy configuration via YAML files for model parameters, training hyperparameters, and dataset classes.
- **Visualization**: Tools for visualizing model predictions and performance metrics.
- **Data Pipeline**: Scripts to download and preprocess subsets of the COCO dataset directly from URLs.

## Project Structure

```
├── configs/
│   └── config.yaml           # Configuration for model, data, and training
├── data/
│   ├── instances_train2017.json # (Required) COCO annotations file
│   └── metadata.json         # Generated metadata
├── notebooks/
│   └── coco_cls.ipynb        # Main notebook for training and evaluation
├── src/
│   ├── data_utils.py         # Dataset loading and processing utilities
│   ├── download_subset_data.py # Script to download COCO subset
│   ├── engine.py             # Evaluation and inference logic
│   ├── models.py             # CLIP + Linear Classifier model definition
│   └── visualization.py      # Plotting and visualization tools
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd Coco_classification
    ```

2. **Install dependencies:**
    This project requires Python 3.8+ and PyTorch 2.0+.

    ```bash
    pip install -r requirements.txt
    ```

    *Note: The `clip` package is installed directly from OpenAI's GitHub repository.*

## Data Preparation

The project is designed to work with a subset of the COCO dataset.

1. **Download Annotations**:
    You need the `instances_train2017.json` file from the [COCO dataset](https://cocodataset.org/). Place it in the root directory or update the path in `src/download_subset_data.py` (variable `ANNOTATION_PATH`).

2. **Download Images**:
    Use the provided script to download a subset of images for the target classes.

    ```bash
    python src/download_subset_data.py
    ```

    This script will:
    - Parse the annotation file.
    - Filter images for the target classes (e.g., person, bicycle, car, dog, etc.).
    - Download and resize images (default: 336x336).
    - Save them to a zip file or directory as configured.

## Configuration

Modify `configs/config.yaml` to adjust experiment settings:

```yaml
data:
  image_size: 336
  batch_size: 32
  classes: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', ...]

model:
  name: "ViT-L/14@336px"
  device: "cuda"

train:
  learning_rate: 0.001
  epochs: 20
  few_shot_k: 16
```

## Usage

The primary workflow is demonstrated in the Jupyter Notebook:

1. Open `notebooks/coco_cls.ipynb`.
2. Run the cells to:
    - Load the configuration.
    - Initialize the CLIP-based model.
    - Train the linear classifier head.
    - Evaluate performance (including Zero-Shot capabilities).
    - Visualize results.

## Model Architecture

The model code in `src/models.py` defines a `CLIPFewShotModel` which:

1. Freezes the pre-trained CLIP parameters (`clip_model`).
2. Extracts image features using `clip_model.encode_image`.
3. Passes features through a trainable `nn.Linear` layer (`classifier`) to output class probabilities.

## License

This project relies on the COCO dataset and OpenAI's CLIP model. Please verify their respective licenses for usage.
