import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class TabularDataset(Dataset):
    def __init__(self, features):
        self.features = torch.FloatTensor(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class TabularModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(TabularModel, self).__init__()
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        
        # Output layer
        layers.append(torch.nn.Linear(hidden_size, 1))
        
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    config = checkpoint['config']
    
    model = TabularModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['scaler'], config

def perform_inference(config):
    # Load model and scaler
    model, scaler, model_config = load_model(config['data']['model_save_path'])
    
    # Load and preprocess data
    df = pd.read_csv(config['data']['inference_path'])
    features = df[config['training']['feature_columns']].values
    
    # Scale features
    features = scaler.transform(features)
    
    # Create dataset and dataloader
    dataset = TabularDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=config['inference']['batch_size'],
        shuffle=False
    )
    
    # Perform inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch)
            predictions.extend(outputs.numpy())
    
    # Save predictions
    predictions = np.array(predictions).flatten()
    df['predictions'] = predictions
    df.to_csv(config['inference']['output_path'], index=False)
    print(f"Predictions saved to {config['inference']['output_path']}")

if __name__ == "__main__":
    # Load configuration
    with open('input_parameters.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(config['inference']['output_path']), exist_ok=True)
    
    # Perform inference
    perform_inference(config) 