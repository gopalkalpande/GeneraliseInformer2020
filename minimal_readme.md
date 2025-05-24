# Tabular Data Training and Inference Pipeline

A PyTorch-based pipeline for training and inference on tabular data.

## Setup

1. Create the required directory structure:
```bash
mkdir -p inputData/{train,test,inference} modelData
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place your training data in `inputData/train/train_data.csv`
2. Place your test data in `inputData/test/test_data.csv`
3. Place your inference data in `inputData/inference/inference_data.csv`

## Configuration

Edit `input_parameters.yml` to configure:
- Data paths
- Model architecture (input size, hidden size, layers, dropout)
- Training parameters (batch size, epochs, early stopping)
- Feature and target column names

Example configuration:
```yaml
data:
  train_path: "inputData/train/train_data.csv"
  test_path: "inputData/test/test_data.csv"
  inference_path: "inputData/inference/inference_data.csv"
  model_save_path: "modelData/trained_model.pth"

model:
  input_size: 10  # Number of features
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
  learning_rate: 0.001
  weight_decay: 1e-5

training:
  batch_size: 32
  num_epochs: 100
  early_stopping_patience: 10
  validation_split: 0.2
  target_column: "target"
  feature_columns:
    - "feature1"
    - "feature2"
    - "feature3"
```

## Training

Run the training script:
```bash
python train.py
```

The trained model will be saved in `modelData/trained_model.pth`.

## Inference

Run the inference script:
```bash
python inference.py
```

Predictions will be saved in `inputData/inference/predictions.csv`.

## Model Architecture

The model consists of:
- Input layer
- Multiple hidden layers with ReLU activation and dropout
- Output layer

The architecture is configurable through the `input_parameters.yml` file. 