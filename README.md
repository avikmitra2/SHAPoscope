# SHAPoscope
Codes and related files
Training file (in 7z compressed) and testing files are also uploaded.

hidden_layer_1_biases_epoch.csv, hidden_layer_2_biases_epoch.csv, etc., are for bias values during training.
hidden_layer_1_weights.csv, hidden_layer_2_weights.csv, etc., are for weight values during training.

hidden_layer_1_shap_values.csv, hidden_layer_3_shap_values.csv, etc., are for SHAP value signifying contributions of features to the neurons of the corresponding layers. Note that there is dropout layer after each hidden layer, so - L1 --> 1, L2 --> 3, L3 --> 5, Output layer --> 7.

Post-training CSV files containing data starts with "post_training_".

feature_frequency_layer1.csv, feature_frequency_layer3.csv, etc., contains the frequency of occurrence of features among all the neurons of a layer, that have maximum SHAPley values within a neuron. Maximum SHAPley value is identified among each neuron of a layer, and then the feature occurrence is observed among all the neurons of the layer. Then the frequency table is created for each epoch of a layer.

grouped_feature_stats_layer1.csv, grouped_feature_stats_layer3.csv, etc., contains trend of descriptive statistics, layer-by-layer (separate file for each layer). Each file contains epoch-by-epoch descriptive statistics for each of the contributing feature (descriptive statistics of the SHAPley values), grouped by the type of the descriptive statistic parameters (for example, all means of the features are grouped together).

The hidden_layer_1_neuron_to_output_shap_descriptive_stats.csv, hidden_layer_3_neuron_to_output_shap_descriptive_stats.csv, hidden_layer_5_neuron_to_output_shap_descriptive_stats.csv files for hidden-layer 1, 2 and 3 respectively, contains the following information:
(1) Epoch-by-epoch, number of neurons with 0 values, negative values and positive values.
(2) Descriptive statistics (Mean, median and mode; Standard Deviation, min, max, quartiles; Skewness & Kurtosis) epoch-by-epoch:
   (2.1) Including the 0 SHAPley values.
   (2.2) excluding the 0 SHAPley values.

Descriptive Statistics for weights and biases for each layer (separate CSV files for weights and biases and layers):
layer 1 - hidden layer1; layer 2 - hidden layer2; layer 3 - hidden layer 3; layer 4 - output layer.
Files: layer_1_weights_stats.csv, layer_1_biases_stats.csv, layer_2_weights_stats.csv, layer_2_biases_stats.csv, etc.

