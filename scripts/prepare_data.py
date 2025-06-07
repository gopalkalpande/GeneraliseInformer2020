import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

# Download the dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='target')

# Combine features and target
df = pd.concat([X, y], axis=1)

# Create directories if they don't exist
os.makedirs('inputData/train', exist_ok=True)
os.makedirs('inputData/test', exist_ok=True)
os.makedirs('inputData/inference', exist_ok=True)

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 50% test, 50% inference from the temp set
test_df, inference_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save datasets
train_df.to_csv('inputData/train/train_data.csv', index=False)
test_df.to_csv('inputData/test/test_data.csv', index=False)
inference_df.to_csv('inputData/inference/inference_data.csv', index=False)

print("Datasets prepared and saved:")
print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")
print(f"Inference set: {len(inference_df)} samples")
print("\nFeature names:")
for name in california.feature_names:
    print(f"- {name}") 