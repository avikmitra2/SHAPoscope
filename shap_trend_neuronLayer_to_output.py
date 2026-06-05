#shap_trend_neuronLayer_to_output.py

import glob
import re
import numpy as np
import pandas as pd

# Find the specific neuron-to-output SHAP contribution CSV files
contribution_files = glob.glob("hidden_layer_*_neuron_to_output_shap.csv")

layer_distances = {}

for file_path in contribution_files:
    match = re.search(r"hidden_layer_(\d+)_neuron_to_output_shap.csv", file_path)
    if not match:
        continue
    layer_num = match.group(1)
    
    # Load and sort by epoch
    df = pd.read_csv(file_path).sort_values('epoch')
    
    # FIX: Drop columns that are entirely NaN to prevent NaN propagation
    df_clean = df.dropna(axis=1, how='all')
    
    # Select clean tracking metric columns
    contribution_cols = [col for col in df_clean.columns if col != 'epoch']
    contribution_matrix = df_clean[contribution_cols].values
    
    # Calculate Euclidean distance step-by-step
    subsequent_diffs = np.diff(contribution_matrix, axis=0)
    euclidean_distances = np.sqrt(np.sum(subsequent_diffs ** 2, axis=1))
    
    layer_distances[f"layer{layer_num}"] = euclidean_distances

# Build final output matrix
sample_layer = list(layer_distances.keys())[0]
num_changes = len(layer_distances[sample_layer])
epochs = np.arange(2, 2 + num_changes)

output_df = pd.DataFrame({'epoch': epochs})
sorted_layers = sorted(layer_distances.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
for layer in sorted_layers:
    output_df[layer] = layer_distances[layer]

# Regenerate output file
output_df.to_csv("neuron_to_output_shap_changes.csv", index=False)
print("Successfully regenerated neuron_to_output_shap_changes.csv with valid layer5 values!")