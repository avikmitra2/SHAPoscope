#shap_trend_feature_to_layer.py

import glob
import re
import numpy as np
import pandas as pd

# Find all uploaded SHAPley value files automatically
shap_files = glob.glob("hidden_layer_*_shap_values.csv")

# Dictionary to store the Euclidean distance metrics per layer
layer_distances = {}

# Process each file found
for file_path in shap_files:
    # Extract the layer number from the filename
    match = re.search(r"hidden_layer_(\d+)_shap_values.csv", file_path)
    if not match:
        continue
    layer_num = match.group(1)
    
    # Load and ensure rows are ordered chronologically by epoch
    df = pd.read_csv(file_path).sort_values('epoch')
    
    # Select all columns representing SHAPley values (exclude 'epoch')
    shap_cols = [col for col in df.columns if col != 'epoch']
    shap_matrix = df[shap_cols].values
    
    # Step 1: Calculate the coordinate-wise difference between subsequent rows
    # (row t minus row t-1)
    subsequent_diffs = np.diff(shap_matrix, axis=0)
    
    # Step 2: Compute Euclidean distance for the entire vector per epoch change
    # distance = sqrt( sum( (x_t - x_{t-1})^2 ) )
    euclidean_distances = np.sqrt(np.sum(subsequent_diffs ** 2, axis=1))
    
    # Store results in the dictionary
    layer_distances[f"layer{layer_num}"] = euclidean_distances

# Determine the number of epoch changes recorded
sample_layer = list(layer_distances.keys())[0]
num_changes = len(layer_distances[sample_layer])

# Generate the 1-indexed epoch column starting from 2
# (epoch 2 signifies the change from epoch 1 to 2)
epochs = np.arange(2, 2 + num_changes)

# Build the final combined DataFrame
output_df = pd.DataFrame({'epoch': epochs})

# Dynamically add the layer columns in sorted numerical order
sorted_layers = sorted(layer_distances.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
for layer in sorted_layers:
    output_df[layer] = layer_distances[layer]

# Save the resulting data to a CSV file
output_df.to_csv("shap_changes.csv", index=False)
print("Successfully generated 'shap_changes.csv'!")