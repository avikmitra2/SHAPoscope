# SHAPoscope
Codes and related files
hidden_layer_1_biases_epoch.csv, hidden_layer_2_biases_epoch.csv, etc., are for bias values during training.
hidden_layer_1_weights.csv, hidden_layer_2_weights.csv, etc., are for weight values during training.
hidden_layer_1_shap_values.csv, hidden_layer_3_shap_values.csv, etc., are for SHAP value signifying contributions of features to the neurons of the corresponding layers. Note that there is dropout layer after each hidden layer, so - L1 --> 1, L2 --> 3, L3 --> 5, Output layer --> 7.
