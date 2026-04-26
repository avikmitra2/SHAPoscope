#NN3.py

#Steps:
#1. Load Data
#2. Define Keras Model
#3. Compile Keras Model
#4. Fit Keras Model
#5. Evaluate Keras Model
#6. Tie It All Together
#7. Make Predictions


# first neural network with keras tutorial
from numpy import loadtxt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#import tensorflow.keras.metrics as metrics
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

import numpy as np
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

from tensorflow.keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTE

from tensorflow.keras.callbacks import Callback

import shap #For explainability using SHAP.

import os

from tensorflow.keras.layers import Input


# load the tranformed TRAINING dataset (binary classification)
training_dataset = pd.read_csv( 'nsl_kdd_training_full_main3.csv' )
print( training_dataset.shape )

# split into input (X) and output (y) variables
X_train = training_dataset.iloc[:, 0 : (training_dataset.shape[1] - 1) ]

y_train = training_dataset.iloc[:, training_dataset.shape[1] - 1 ]

print( f"\nTraining dataset shape: {training_dataset.shape}" )

smote = SMOTE(sampling_strategy='auto', random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"Resampled training shape: {X_train.shape}")

# Load the transformed TEST dataset (binary classification)

test_dataset = pd.read_csv( 'nsl_kdd_testing_full_main3.csv' )
print( f"\n Test dataset shape: {test_dataset.shape}" )

# split into input (X) and output (y) variables
X_test = test_dataset.iloc[:, 0 : (test_dataset.shape[1] - 1) ]

y_test = test_dataset.iloc[:, test_dataset.shape[1] - 1 ]

#Code for setting up the ANN

print("Checking data readiness...")
print(f"X_train type: {type(X_train)}, Shape: {X_train.shape}")

nodes = 126 #Number of nodes per hidden layers
function = 'relu' #Activation function in input layer and hidden layer.
loss_func = 'binary_crossentropy'
#opt = 'adam'
opt = Adam(learning_rate=0.001) #Changed from 0.0001

epoch_count = 100
batch_size_count = 128
dropout_rate = 0.1
l2_regualization = 0.01

weights = {0: 1.0, 1: 1.5}

early_stop = EarlyStopping(
    monitor='val_loss',      # What metric to watch
    patience=10,              # How many epochs to wait for improvement before stopping
    restore_best_weights=True, # VERY IMPORTANT: Reverts model to its best state
    verbose=1
)

#For accessing weights at each epoch.

np.set_printoptions(threshold=np.inf)

class WeightCapture(Callback):
     
    def on_train_begin(self, logs=None):
        """Runs once at the start of model.fit() to create files and headers."""
        weight_layer_count = 1
        for layer in self.model.layers:
            weights_list = layer.get_weights()
            if len(weights_list) > 0:
                weights = weights_list[0]
                num_inputs = weights.shape[0]  # Rows (Input features)
                num_neurons = weights.shape[1] # Columns (Neurons in this layer)
                
                filename = f"hidden_layer_{weight_layer_count}_weights.csv"
                
                # Create specific column names:
                # Format: neuron1_connection1, neuron1_connection2 ... neuron2_connection1
                headers = ["epoch_number"]
                for n in range(num_neurons):
                    for i in range(num_inputs):
                        headers.append(f"neuron{n+1}_connection{i+1}")
                
                # Write header to new file
                with open(filename, "w") as f:
                    f.write(",".join(headers) + "\n")
                
                print(f"Created {filename} with columns mapped to {num_neurons} neurons.")
                

                #For noting the biases.

                b_filename = f"hidden_layer_{weight_layer_count}_biases_epoch.csv"
                b_headers = ["epoch"] + [f"neuron{n+1}_bias" for n in range(num_neurons)]
                with open(b_filename, "w") as f:
                    f.write(",".join(b_headers) + "\n")

                print(f"Created {b_filename} with columns mapped to {num_neurons} neurons.")

                weight_layer_count += 1

                               

    def on_epoch_end(self, epoch, logs=None):
        """Runs at the end of every epoch to append the current weights."""
        weight_layer_count = 1 
        for layer in self.model.layers:
            weights_list = layer.get_weights()
            
            if len(weights_list) > 0:
                weights, biases = weights_list[0], weights_list[1]
                
                # Flatten the matrix into a 1D array of all connections
                flattened_weights = weights.ravel()

                flattened_biases = biases.ravel()
                
                # Convert all weight values to strings with 6 decimal places
                str_weights = [format(w, ".6f") for w in flattened_weights]

                str_biases = [format(b, ".6f") for b in flattened_biases]
                
                # Format line as: epoch_number,w1,w2,w3...
                csv_line = f"{epoch}," + ",".join(str_weights) + "\n"

                csv_line_bias = f"{epoch}," + ",".join(str_biases) + "\n"
                
                filename = f"hidden_layer_{weight_layer_count}_weights.csv"

                filename_bias = f"hidden_layer_{weight_layer_count}_biases_epoch.csv"

                with open(filename, "a") as f:
                    f.write(csv_line)

                with open(filename_bias, "a") as f:
                    f.write(csv_line_bias)
                
                weight_layer_count += 1

#End of callback WeightCapture class.

#Class for explanation using SHAP.

class ShapCapture(Callback):
    def __init__(self, background, test_subset, feature_names):
        super().__init__()
        self.background = background
        self.test_subset = test_subset
        self.feature_names = feature_names

    def on_train_begin(self, logs=None):
        self.layer_indices = [i for i, layer in enumerate(self.model.layers) if isinstance(layer, tf.keras.layers.Dense)]
        self.hidden_indices = self.layer_indices[:-1]
        self.neuron_to_output_files = {}

        for idx in self.layer_indices:
            num_neurons = self.model.layers[idx].units

            filename = f"hidden_layer_{idx+1}_shap_values.csv"
            headers = ["epoch"]
            for n in range(num_neurons):
                for feat in self.feature_names:
                    headers.append(f"neuron{n+1}_{feat}")
            with open(filename, "w") as f:
                f.write(",".join(headers) + "\n")

            # --- NEW NEURON-TO-OUTPUT FILE (Added logic) ---
            # We only track this for hidden layers
            if idx in self.hidden_indices:
                 out_filename = f"hidden_layer_{idx+1}_neuron_to_output_shap.csv"
                 self.neuron_to_output_files[idx] = out_filename
                 headers = ["epoch"] + [f"neuron{i+1}_contribution" for i in range(num_neurons)]
            with open(out_filename, "w") as f:
               f.write(",".join(headers) + "\n")


    def on_epoch_end(self, epoch, logs=None):
        """Calculates SHAP values and handles the data structure safely to avoid IndexError."""
        for idx in self.layer_indices:
            num_neurons = self.model.layers[idx].units

            filename = f"hidden_layer_{idx+1}_shap_values.csv"
            
            # FIX WARNING: Pass the tensor directly (not in a list)
            intermediate_model = tf.keras.Model(
                inputs=self.model.inputs[0], 
                outputs=self.model.layers[idx].output
            )
            
            explainer = shap.GradientExplainer(intermediate_model, self.background)
            
            try:
                # Calculate SHAP values for the layer
                all_neuron_shap = explainer.shap_values(self.test_subset)
                
                # FIX INDEXERROR: Ensure data is in a list where each element is one neuron
                # GradientExplainer sometimes returns a single array (samples, features, neurons)
                # We need to swap axes so we can iterate by neuron
                if isinstance(all_neuron_shap, np.ndarray):
                    # Transpose to (neurons, samples, features)
                    all_neuron_shap = np.transpose(all_neuron_shap, (2, 0, 1))
                elif isinstance(all_neuron_shap, list) and len(all_neuron_shap) == 1:
                    # If it's a list containing one 3D array
                    all_neuron_shap = np.transpose(all_neuron_shap[0], (2, 0, 1))
                    
            except Exception as e:
                print(f"Error calculating SHAP for layer {idx+1}: {e}")
                continue

            with open(filename, "a", buffering=1) as f:
                f.write(f"{epoch}")
                
                # Now we can safely iterate up to num_neurons
                for n_idx in range(num_neurons):
                    try:
                        neuron_shap = all_neuron_shap[n_idx]
                        
                        # Calculate mean absolute importance (1D array of features)
                        mean_importance = np.abs(neuron_shap).mean(axis=0).ravel()
                        
                        for val in mean_importance:
                            # Use .item() for safe scalar conversion
                            f.write(f",{val.item():.6f}")
                        
                        f.flush()
                    except IndexError:
                        print(f"Warning: Could not find SHAP data for neuron {n_idx} in layer {idx+1}")
                        break
                
                f.write("\n")
                os.fsync(f.fileno())

            # --- NEW NEURON-TO-OUTPUT LOGIC ---
            if idx in self.hidden_indices:
                # Build sub-model from CURRENT layer to the END
                hidden_input = Input(shape=(num_neurons,))
                curr_output = hidden_input
                for i in range(idx + 1, len(self.model.layers)):
                    curr_output = self.model.layers[i](curr_output)
                
                ntoo_model = tf.keras.Model(inputs=hidden_input, outputs=curr_output)

                # Get activations to use as input features for SHAP
                act_model = tf.keras.Model(inputs=self.model.inputs[0], outputs=self.model.layers[idx].output)
                h_act_bg = act_model.predict(self.background, verbose=0)
                h_act_test = act_model.predict(self.test_subset, verbose=0)

                # Calculate impact of these neurons on final binary output
                ntoo_explainer = shap.GradientExplainer(ntoo_model, h_act_bg)
                ntoo_shap = ntoo_explainer.shap_values(h_act_test)
                if isinstance(ntoo_shap, list): ntoo_shap = ntoo_shap[0]

                mean_impact = np.abs(ntoo_shap).mean(axis=0).flatten()

                # Write to the specific layer file
                with open(self.neuron_to_output_files[idx], "a") as f:
                    f.write(f"{epoch}," + ",".join([f"{v.item():.6f}" for v in mean_impact]) + "\n")


#End of class ShapCapture's defintion


# Prepare SHAP data (ensure float32 for GradientExplainer)
shap_bg = X_train.values[np.random.choice(X_train.shape[0], 100, replace=False)].astype('float32')
shap_test = X_test.values[np.random.choice(X_test.shape[0], 50, replace=False)].astype('float32')

# Initialize the new callback
shap_callback = ShapCapture(shap_bg, shap_test, X_test.columns) #Should be done at least before fit() call (training phase)

ann_model = Sequential()

ann_model.add(Input(shape=(training_dataset.shape[1]-1,))) ; ann_model.add(Dense(nodes, activation = function, kernel_regularizer=regularizers.l2(l2_regualization))) #1st Hidden layer and NOT Input layer.
ann_model.add(Dropout( dropout_rate ))

ann_model.add(Dense(64, activation = function, kernel_regularizer=regularizers.l2( l2_regualization ) )) #2nd hidden layer
ann_model.add(Dropout( dropout_rate ))

ann_model.add(Dense(32, activation = function, kernel_regularizer=regularizers.l2( l2_regualization ) )) #3rd hidden layer
ann_model.add(Dropout( dropout_rate ))

ann_model.add(Dense(1, activation='sigmoid')) #Output-Layer

#ann_model.compile(loss = loss_func, optimizer= opt, metrics = [ metrics.BinaryAccuracy(), metrics.FalseNegatives(),  
#                                                                        metrics.FalsePositives(), metrics.TrueNegatives(), 
#                                                                        metrics.TruePositives() ] )

ann_model.compile(loss = loss_func, optimizer= opt, metrics = [ metrics.BinaryAccuracy() ] )

#Grid Search for hyper-parameter setting

# <<<Separate Programme File >>>

weight_callback = WeightCapture()

#history = ann_model.fit(X_train, y_train, epochs=epoch_count, batch_size=batch_size_count, validation_split=0.1, callbacks=[early_stop], verbose = 1)  #, use_multiprocessing = True) 
history = ann_model.fit(X_train, y_train, epochs=epoch_count, batch_size=batch_size_count, validation_data=(X_test, y_test), callbacks=[weight_callback, shap_callback], verbose = 1)  #, use_multiprocessing = True)
#Early stop removed.

results = ann_model.evaluate(X_test, y_test, batch_size=batch_size_count, verbose = 1) #, use_multiprocessing = True)


################ Post-training weight and bias value ################################

print("Saving final post-training parameters...")
layer_count = 1
for layer in ann_model.layers:
    params = layer.get_weights()
    if len(params) > 0:
        weights, biases = params[0], params[1]
        
        # 1. Save Post-Training Weights
        # Rows = Neurons, Columns = Input Connections
        w_df = pd.DataFrame(weights.T) 
        w_df.columns = [f"input_{i+1}" for i in range(weights.shape[0])]
        w_df.index = [f"neuron_{i+1}" for i in range(weights.shape[1])]
        w_df.to_csv(f"post_training_layer_{layer_count}_weights.csv")
        
        # 2. Save Post-Training Biases
        b_df = pd.DataFrame(biases, columns=["bias_value"])
        b_df.index = [f"neuron_{i+1}" for i in range(len(biases))]
        b_df.to_csv(f"post_training_layer_{layer_count}_biases.csv")
        
        print(f"Layer {layer_count} parameters exported.")
        layer_count += 1



################# Explanation using SHAP (after training) ############################

import shap
import matplotlib.pyplot as plt

# 1. Use NumPy arrays (ensure they are float32)
background = X_train.values[np.random.choice(X_train.shape[0], 100, replace=False)].astype('float32')
X_test_subset = X_test.values[np.random.choice(X_test.shape[0], 100, replace=False)].astype('float32')

# 2. Use GradientExplainer instead of DeepExplainer
explainer = shap.GradientExplainer(ann_model, background)

# 3. Calculate SHAP values
# GradientExplainer returns a list of arrays for Keras models
shap_values = explainer.shap_values(X_test_subset)

#For storing weights of SHAP

shap_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

# Calculate the mean absolute SHAP value for each feature across the subset
global_mean_shap = np.abs(shap_to_plot).mean(axis=0).flatten()

# Create a DataFrame to easily handle the mapping and CSV export
post_training_shap_df = pd.DataFrame([global_mean_shap], columns=X_test.columns)

# Save to a unique filename to avoid clashing with hidden layer CSVs
post_training_filename = "post_training_global_shap_importance.csv"
post_training_shap_df.to_csv(post_training_filename, index=False)
print(f"Post-training SHAP importance saved to: {post_training_filename}")
# --------------------------------------------------------

# 4. Visualization
shap.summary_plot(shap_to_plot, X_test_subset, feature_names=X_test.columns)

# 5. Local Explanation
sample_idx = 0

# 5.1. Manually calculate the base value (expected value) 
# GradientExplainer requires this manual step in many environments
base_val = ann_model.predict(background).mean()

# 5.2. Correctly slice the SHAP values
# shap_values[0] is (100, 41). We need one row: (41,)
# We use .flatten() or [sample_idx] to ensure it is 1D
single_shap_record = shap_to_plot[sample_idx].flatten()

# 5.3. Correctly slice the Feature values
# X_test_subset is (100, 41). We need one row: (41,)
single_feature_record = X_test_subset[sample_idx]

shap.force_plot(
    base_val, 
    single_shap_record, 
    single_feature_record, 
    feature_names=X_test.columns, 
    matplotlib=True,
    show=True
)


### Feature to Neuron importance using SHAP and Neuron importance to output using SHAP #####

# Define indices for all dense layers
dense_indices = [i for i, layer in enumerate(ann_model.layers) if isinstance(layer, Dense)]
hidden_indices = dense_indices[:-1]  # All layers except the final output

for idx in dense_indices:
    num_neurons = ann_model.layers[idx].units
    
    # --- (1) Feature Importance per Neuron (Post-Training) ---
    # Model: Raw Inputs -> Specific Hidden Layer Activations
    feat_to_neuron_model = tf.keras.Model(inputs=ann_model.inputs[0], outputs=ann_model.layers[idx].output)
    explainer_feat = shap.GradientExplainer(feat_to_neuron_model, background)
    shap_feat = explainer_feat.shap_values(X_test_subset)
    
    # Standardize shape to (neurons, samples, features)
    if isinstance(shap_feat, np.ndarray):
        shap_feat = np.transpose(shap_feat, (2, 0, 1))
    elif isinstance(shap_feat, list) and len(shap_feat) == 1:
        shap_feat = np.transpose(shap_feat[0], (2, 0, 1))
    
    # Save Feature-to-Neuron importance to a CSV for this layer
    fn_filename = f"post_training_layer_{idx+1}_feature_to_neuron_importance.csv"
    fn_data = []
    for n in range(num_neurons):
        mean_abs_feat = np.abs(shap_feat[n]).mean(axis=0)
        fn_data.append(mean_abs_feat)
    
    # Each row in this CSV represents a neuron; each column is a feature
    pd.DataFrame(fn_data, columns=X_test.columns).to_csv(fn_filename, index_label="neuron_index")
    print(f"Saved Post-Training Feature-to-Neuron: {fn_filename}")

    # --- (2) Neuron Importance to Binary Output (Post-Training) ---
    if idx in hidden_indices:
        # Model: Hidden Activations -> Final Model Output
        hidden_input = Input(shape=(num_neurons,))
        curr_logic = hidden_input
        for i in range(idx + 1, len(ann_model.layers)):
            curr_logic = ann_model.layers[i](curr_logic)
        
        neuron_to_out_model = tf.keras.Model(inputs=hidden_input, outputs=curr_logic)
        
        # Get activations of this layer to serve as 'features' for the explainer
        act_model = tf.keras.Model(inputs=ann_model.inputs[0], outputs=ann_model.layers[idx].output)
        h_act_bg = act_model.predict(background, verbose=0)
        h_act_test = act_model.predict(X_test_subset, verbose=0)
        
        explainer_out = shap.GradientExplainer(neuron_to_out_model, h_act_bg)
        shap_out = explainer_out.shap_values(h_act_test)
        if isinstance(shap_out, list): shap_out = shap_out[0]
        
        # Save Neuron-to-Output importance to a CSV for this layer
        no_filename = f"post_training_layer_{idx+1}_neuron_to_output_importance.csv"
        mean_abs_neuron = np.abs(shap_out).mean(axis=0).flatten()
        
        # Single row representing the global importance of every neuron in this layer
        pd.DataFrame([mean_abs_neuron], columns=[f"neuron_{i+1}" for i in range(num_neurons)]).to_csv(no_filename, index=False)
        print(f"Saved Post-Training Neuron-to-Output: {no_filename}")
