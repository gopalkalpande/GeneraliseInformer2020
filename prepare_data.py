import pandas as pd
import numpy as np
import os
import requests
from zipfile import ZipFile
from io import BytesIO

def download_and_prepare_data():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    print("Downloading dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    response = requests.get(url)
    
    # Extract the zip file
    with ZipFile(BytesIO(response.content)) as zip_file:
        # Read the data
        df = pd.read_csv(zip_file.open('household_power_consumption.txt'), 
                        sep=';', 
                        parse_dates={'datetime': ['Date', 'Time']},
                        na_values=['?', 'NA'])
    
    # Handle missing values
    df = df.fillna(method='ffill')
    
    # Convert numeric columns
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 
                      'Voltage', 'Global_intensity', 
                      'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Create train and test splits (last 30 days for testing)
    test_size = 30 * 24 * 60  # 30 days worth of minutes
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    
    # Save the datasets
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("Data preparation complete!")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Print column information
    print("\nAvailable features:")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    download_and_prepare_data() 