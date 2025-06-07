import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from train import TabularModel, TabularDataset
from sklearn.metrics import mean_squared_error, r2_score

class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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

def load_model(model_path, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = TabularModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Load saved model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
    
    return model, scaler, device

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    return mse, r2, predictions

def main():
    # Load configuration
    with open('input_parameters.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get timestamp from config and update paths
    timestamp = config['data']['timestamp']
    model_path = config['data']['model_save_path'].replace('{{timestamp}}', timestamp)
    output_path = config['inference']['output_path'].replace('{{timestamp}}', timestamp)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model and scaler
    model, scaler, device = load_model(model_path, config)
    
    # Evaluate on test set
    test_df = pd.read_csv(config['data']['test_path'])
    X_test = test_df[config['training']['feature_columns']].values
    y_test = test_df[config['training']['target_column']].values
    
    X_test_scaled = scaler.transform(X_test)
    test_dataset = TabularDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config['inference']['batch_size'], shuffle=False)
    
    test_mse, test_r2, test_predictions = evaluate_model(model, test_loader, device)
    
    print("\nTest Set Results:")
    print(f"Mean Squared Error: {test_mse:.4f}")
    print(f"R² Score: {test_r2:.4f}")
    
    # Evaluate on inference set
    inference_df = pd.read_csv(config['data']['inference_path'])
    X_inf = inference_df[config['training']['feature_columns']].values
    y_inf = inference_df[config['training']['target_column']].values
    
    X_inf_scaled = scaler.transform(X_inf)
    inference_dataset = TabularDataset(X_inf_scaled, y_inf)
    inference_loader = DataLoader(inference_dataset, batch_size=config['inference']['batch_size'], shuffle=False)
    
    inf_mse, inf_r2, inf_predictions = evaluate_model(model, inference_loader, device)
    
    print("\nInference Set Results:")
    print(f"Mean Squared Error: {inf_mse:.4f}")
    print(f"R² Score: {inf_r2:.4f}")
    
    # Save predictions
    inference_df['predicted_target'] = inf_predictions
    inference_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

if __name__ == "__main__":
    main() 