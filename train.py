import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from models.model import Informer
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, features, target, time_col):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.time_col = time_col
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # Get feature sequences
        seq_x = self.data[s_begin:s_end][self.features].values
        seq_y = self.data[r_begin:r_end][self.target].values
        
        # Get time features - ensure we're working with datetime objects
        time_stamps_x = pd.DatetimeIndex(self.data[s_begin:s_end][self.time_col])
        time_features_x = time_features(time_stamps_x, freq='h')
        
        time_stamps_y = pd.DatetimeIndex(self.data[r_begin:r_end][self.time_col])
        time_features_y = time_features(time_stamps_y, freq='h')
        
        return {
            'x_enc': torch.FloatTensor(seq_x),
            'x_mark_enc': torch.FloatTensor(time_features_x),
            'x_dec': torch.FloatTensor(seq_y),
            'x_mark_dec': torch.FloatTensor(time_features_y),
            'y': torch.FloatTensor(seq_y[-self.pred_len:])
        }

def create_informer_model(config, device):
    model = Informer(
        enc_in=config['model']['enc_in'],
        dec_in=config['model']['dec_in'],
        c_out=config['model']['c_out'],
        seq_len=config['model']['seq_len'],
        label_len=config['model']['label_len'],
        out_len=config['model']['out_len'],
        factor=config['model']['factor'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        e_layers=config['model']['e_layers'],
        d_layers=config['model']['d_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        attn=config['model']['attn'],
        embed=config['model']['embed'],
        freq=config['model']['freq'],
        activation=config['model']['activation'],
        output_attention=config['model']['output_attention'],
        distil=config['model']['distil'],
        mix=config['model']['mix'],
        device=device
    )
    return model

def train_model(config):
    print("\n" + "="*50)
    print("STARTING TRAINING PROCESS")
    print("="*50)
    
    # Check device availability
    print("\nChecking device availability...")
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon (MPS) is available!")
        device = torch.device('mps')
        # Don't set default tensor type for MPS
    elif torch.cuda.is_available():
        print("✓ CUDA is available!")
        print(f"  - CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("! No GPU acceleration available. Using CPU instead.")
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    
    print(f"\nUsing device: {device}")
    
    print("\n1. Loading and preprocessing data...")
    df = pd.read_csv(config['data']['train_path'])
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n2. Converting time column to datetime...")
    df[config['training']['time_column']] = pd.to_datetime(df[config['training']['time_column']])
    print(f"✓ Time column converted. Sample timestamps:")
    print(df[config['training']['time_column']].head())
    
    print("\n3. Splitting data into train and validation...")
    train_size = int(len(df) * (1 - config['training']['validation_split']))
    train_df = df[:train_size]
    val_df = df[train_size:]
    print(f"✓ Data split complete:")
    print(f"  - Train set size: {len(train_df)}")
    print(f"  - Validation set size: {len(val_df)}")
    
    print("\n4. Creating datasets...")
    print("Creating training dataset...")
    train_dataset = TimeSeriesDataset(
        train_df,
        config['model']['seq_len'],
        config['model']['label_len'],
        config['model']['out_len'],
        config['training']['feature_columns'],
        [config['training']['target_column']],
        config['training']['time_column']
    )
    print("Creating validation dataset...")
    val_dataset = TimeSeriesDataset(
        val_df,
        config['model']['seq_len'],
        config['model']['label_len'],
        config['model']['out_len'],
        config['training']['feature_columns'],
        [config['training']['target_column']],
        config['training']['time_column']
    )
    print(f"✓ Datasets created successfully")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    
    print("\n5. Creating dataloaders...")
    # Configure DataLoader based on device
    pin_memory = device.type == 'cuda'  # Only use pin_memory for CUDA
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )
    print(f"✓ Dataloaders created successfully")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    
    print("\n6. Initializing model...")
    model = create_informer_model(config, device).to(device)
    
    # Configure device-specific optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    elif device.type == 'mps':
        # Enable Metal performance optimizations if available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.enable_fallback_to_cpu = True
    
    print("Model architecture:")
    print(model)
    print(f"✓ Model initialized successfully")
    
    print("\n7. Setting up optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss().to(device)
    print(f"✓ Optimizer and loss function configured")
    
    print("\n8. Starting training loop...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(config['training']['num_epochs']):
            print(f"\n{'='*20} Epoch {epoch+1}/{config['training']['num_epochs']} {'='*20}")
            
            # Training phase
            model.train()
            train_loss = 0
            print("\nTraining phase:")
            for batch_idx, batch in enumerate(train_loader):
                if batch_idx % 10 == 0:
                    print(f"  Processing batch {batch_idx}/{len(train_loader)}")
                
                # Move data to device
                x_enc = batch['x_enc'].to(device)
                x_mark_enc = batch['x_mark_enc'].to(device)
                x_dec = batch['x_dec'].to(device)
                x_mark_dec = batch['x_mark_dec'].to(device)
                y = batch['y'].to(device)
                
                if batch_idx == 0:
                    print(f"\n  Input shapes:")
                    print(f"  - x_enc: {x_enc.shape}")
                    print(f"  - x_mark_enc: {x_mark_enc.shape}")
                    print(f"  - x_dec: {x_dec.shape}")
                    print(f"  - x_mark_dec: {x_mark_dec.shape}")
                    print(f"  - y: {y.shape}")
                    print(f"  - Device: {x_enc.device}")
                
                optimizer.zero_grad()
                outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(outputs, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Validation phase
            print("\nValidation phase:")
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx % 10 == 0:
                        print(f"  Processing validation batch {batch_idx}/{len(val_loader)}")
                    
                    # Move data to device
                    x_enc = batch['x_enc'].to(device)
                    x_mark_enc = batch['x_mark_enc'].to(device)
                    x_dec = batch['x_dec'].to(device)
                    x_mark_dec = batch['x_mark_dec'].to(device)
                    y = batch['y'].to(device)
                    
                    outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    val_loss += criterion(outputs, y).item()
            
            # Print epoch results
            avg_train_loss = train_loss/len(train_loader)
            avg_val_loss = val_loss/len(val_loader)
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Training Loss: {avg_train_loss:.4f}")
            print(f"  Validation Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print("\nSaving best model...")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, config['data']['model_save_path'])
                print(f"✓ Model saved to {config['data']['model_save_path']}")
            else:
                patience_counter += 1
                print(f"\nNo improvement in validation loss. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
                if patience_counter >= config['training']['early_stopping_patience']:
                    print("Early stopping triggered")
                    break
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model state...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }, config['data']['model_save_path'] + '.interrupted')
        print("Model state saved. You can resume training later.")
        return
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Full error traceback:")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    print("\n" + "="*50)
    print("INITIALIZING TRAINING SCRIPT")
    print("="*50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    if torch.backends.mps.is_available():
        # Set MPS-specific seed if available
        torch.mps.manual_seed(42)
    
    print("\n1. Loading configuration...")
    with open('input_parameters.yml', 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded successfully")
    
    print("\n2. Processing configuration...")
    timestamp = config['data']['timestamp']
    config['data']['model_save_path'] = config['data']['model_save_path'].replace('{{timestamp}}', timestamp)
    config['inference']['output_path'] = config['inference']['output_path'].replace('{{timestamp}}', timestamp)
    print(f"✓ Timestamp processed: {timestamp}")
    
    print("\n3. Creating necessary directories...")
    os.makedirs(os.path.dirname(config['data']['model_save_path']), exist_ok=True)
    print(f"✓ Directories created")
    
    print("\n4. Starting model training...")
    train_model(config) 