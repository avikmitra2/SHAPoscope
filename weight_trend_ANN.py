#weight_trend_ANN.py

import pandas as pd
import numpy as np

# Dictionary to store the distance array for each layer
layer_distances = {}

# Loop through each of the 4 hidden layers
for i in range(1, 5):
    file_name = f'hidden_layer_{i}_weights.csv'
    
    # Load data and ensure it's sorted chronologically by epoch
    df = pd.read_csv(file_name).sort_values('epoch_number')
    
    # Select only the columns representing the weight connections
    weight_cols = [col for col in df.columns if col != 'epoch_number']
    weights_matrix = df[weight_cols].values
    
    # Compute the Euclidean distance between each subsequent pair of weight vectors:
    # d = sqrt( sum( (W_t - W_{t-1})^2 ) )
    distances = np.sqrt(np.sum(np.diff(weights_matrix, axis=0) ** 2, axis=1))
    
    # Store results
    layer_distances[f'layer{i}'] = distances

# Create the 1-indexed epoch column starting from 2
# (epoch 2 signifies the change from epoch 1 to 2)
num_changes = len(layer_distances['layer1'])
epochs = np.arange(2, 2 + num_changes)

# Assemble the final summary DataFrame
output_df = pd.DataFrame({'epoch': epochs})
for i in range(1, 5):
    output_df[f'layer{i}'] = layer_distances[f'layer{i}']

# Save the final results to a new CSV file
output_df.to_csv('weight_changes.csv', index=False)
print("Successfully generated 'weight_changes.csv'!")