# Time Series Forecasting with Informer Model

A PyTorch-based implementation of the Informer model for time series forecasting, with support for multi-variate time series data.

## Setup

1. Create the required directory structure:
```bash
mkdir -p data inputData/{train,test,inference} models/saved_models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Prepare your time series data with the following columns:
   - `datetime`: Timestamp column
   - `Global_active_power`: Target variable
   - `Global_reactive_power`: Feature
   - `Voltage`: Feature
   - `Global_intensity`: Feature
   - `Sub_metering_1`: Feature
   - `Sub_metering_2`: Feature
   - `Sub_metering_3`: Feature

2. Place your data files:
   - Training data: `data/train.csv`
   - Test data: `data/test.csv`
   - Inference data: `inputData/inference/inference_data.csv`

## Configuration

Edit `input_parameters.yml` to configure your model and training:

```yaml
# Data paths
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  inference_path: "inputData/inference/inference_data.csv"
  timestamp: "20240320"  # Update this for each training run
  model_save_path: "models/saved_models/model_{{timestamp}}.pth"

# Model parameters
model:
  # Input/Output dimensions
  enc_in: 7  # Number of input features
  dec_in: 1  # Number of target features
  c_out: 1   # Number of output features
  seq_len: 96    # Input sequence length
  label_len: 48  # Start token length
  out_len: 24    # Prediction sequence length
  
  # Model architecture
  factor: 5
  d_model: 512
  n_heads: 8
  e_layers: 2
  d_layers: 1
  d_ff: 2048
  dropout: 0.05
  attn: "prob"
  embed: "timeF"
  freq: "h"
  activation: "gelu"
  output_attention: false
  distil: true
  mix: true

# Training parameters
training:
  batch_size: 128  # Adjust based on your GPU memory
  num_epochs: 100
  validation_split: 0.1
  early_stopping_patience: 10
  learning_rate: 0.001
  time_column: "datetime"
  feature_columns:
    - "Global_active_power"
    - "Global_reactive_power"
    - "Voltage"
    - "Global_intensity"
    - "Sub_metering_1"
    - "Sub_metering_2"
    - "Sub_metering_3"
  target_column: "Global_active_power"

# Inference parameters
inference:
  batch_size: 32
  output_path: "data/predictions_{{timestamp}}.csv"
```

## Training

1. Run the training script:
```bash
python train.py
```

The training process will:
- Save checkpoints after every epoch in `models/saved_models/model_20240320_epoch_X.pth`
- Save the best model (based on validation loss) in `models/saved_models/model_20240320.pth`
- Implement early stopping if validation loss doesn't improve for 10 epochs

## Model Architecture

The Informer model consists of:
- Encoder with ProbAttention mechanism
- Decoder with cross-attention
- Time series embedding with time features
- Multi-head attention layers
- Feed-forward networks

Key features:
- Probabilistic attention for efficient computation
- Time series specific embeddings
- Distillation mechanism for faster training
- Configurable sequence lengths and prediction horizons

## Inference

Run the inference script:
```bash
python inference.py
```

Predictions will be saved in `data/predictions_20240320.csv`.

## GPU Requirements

- Minimum 8GB GPU memory recommended
- CUDA support required
- Adjust batch size in `input_parameters.yml` based on your GPU memory

## Monitoring Training

To monitor GPU usage during training:
```bash
watch -n 1 nvidia-smi
```

## Model Checkpoints

The training process saves:
1. Epoch checkpoints: `model_20240320_epoch_X.pth`
   - Contains model state, config, epoch number, and losses
   - Saved after every epoch
   - Useful for resuming training or analyzing model progression

2. Best model: `model_20240320.pth`
   - Contains the model state with best validation loss
   - Used for inference
   - Updated only when validation loss improves 