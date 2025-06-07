import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TabularModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(TabularModel, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_data(file_path, feature_columns, target_column):
    df = pd.read_csv(file_path)
    X = df[feature_columns].values
    y = df[target_column].values
    return X, y

def train_model(config):
    # Load and preprocess data
    X, y = load_data(config['data']['train_path'], 
                    config['training']['feature_columns'],
                    config['training']['target_column'])
    
    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['training']['validation_split'], random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'],
                            shuffle=True)
    val_loader = DataLoader(val_dataset,
                          batch_size=config['training']['batch_size'],
                          shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabularModel(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay']
    )
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets.unsqueeze(1)).item()
        
        # Print progress
        print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'config': config
            }, config['data']['model_save_path'])
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    # Load configuration
    with open('input_parameters.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Use the timestamp as provided in the YAML file
    timestamp = config['data']['timestamp']
    config['data']['model_save_path'] = config['data']['model_save_path'].replace('{{timestamp}}', timestamp)
    config['inference']['output_path'] = config['inference']['output_path'].replace('{{timestamp}}', timestamp)
    
    # Create necessary directories
    os.makedirs(os.path.dirname(config['data']['model_save_path']), exist_ok=True)
    
    # Train model
    train_model(config) 