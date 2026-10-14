import pandas as pd
import numpy as np

# 1. Load the SHAP values and attribute map
shap_df = pd.read_csv('hidden_layer_7_shap_values.csv')
attr_df = pd.read_csv('attribute_map2.csv')

# Manual mapping dictionary for variations / one-hot encoded features mapped back to attribute labels
manual_mapping = {
    'dst_host_count': 'dst_hst_count',
    'dst_host_diff_srv_rate': 'dst_hst_diff_srv_rate',
    'dst_host_rerror_rate': 'dst_hst_error_rate',
    'dst_host_same_src_port_rate': 'dst_hst_same_src_port_rate',
    'dst_host_same_srv_rate': 'dst_hst_same_srv_rate',
    'dst_host_serror_rate': 'dst_hst_serror_rate',
    'dst_host_srv_count': 'dst_hst_srv_count',
    'dst_host_srv_diff_host_rate': 'dst_hst_srv_diff_host_rate',
    'dst_host_srv_rerror_rate': 'dst_hst_srv_rerror_rate',
    'dst_host_srv_serror_rate': 'dst_hst_srv_serror_rate',
    'is_guest_login_0': 'is_gst_log',
    'is_guest_login_1': 'is_gst_log',
    'is_host_login_0': 'is_hst_log',
    'is_host_login_1': 'is_hst_log',
    'num_failed_logins': 'nfl',
    'num_file_creations': 'num_file_create',
    'num_outbound_cmds': 'num_otbnd_cmds',
    'serror_rate': 'serr_rate',
    'srv_diff_host_rate': 'srv_diff_hst_rate',
    'srv_serror_rate': 'srv_serr_rate',
    'su_attempted': 'su_atmpt',
    'wrong_fragment': 'wrng_frag',
    'land_0': 'land',
    'land_1': 'land',
    'logged_in_0': 'logged_in',
    'logged_in_1': 'logged_in',
    'num_access_files': 'num_access_file',
    'num_compromised': 'num_com',
    'num_shells': 'num_shell',
    'urgent': 'urg',
    'dst_host_srv_diff_host_rate': 'dst_hst_srv_diff_host_rate'
}

def get_attr_category(col_name):
    # Step 1: Remove "neuron1_" prefix first as requested
    clean_name = col_name.replace('neuron1_', '')
    
    # Step 2: Check manual mapping or direct match in attribute labels
    if clean_name in manual_mapping:
        lbl = manual_mapping[clean_name]
        match = attr_df[attr_df['attr_lbl'] == lbl]
        if not match.empty:
            return match.iloc[0]['attr_category']
            
    match = attr_df[attr_df['attr_lbl'] == clean_name]
    if not match.empty:
        return match.iloc[0]['attr_category']
        
    # Step 3: Check prefixes for one-hot encoded categorical features
    if clean_name.startswith('protocol_'):
        match = attr_df[attr_df['attr_lbl'] == 'protocol']
        if not match.empty: return match.iloc[0]['attr_category']
    if clean_name.startswith('service_'):
        match = attr_df[attr_df['attr_lbl'] == 'service']
        if not match.empty: return match.iloc[0]['attr_category']
    if clean_name.startswith('flag_'):
        match = attr_df[attr_df['attr_lbl'] == 'flag']
        if not match.empty: return match.iloc[0]['attr_category']
            
    return 'unknown'

# 2. Process epoch by epoch
feature_cols = [c for c in shap_df.columns if c != 'epoch']
output_rows = []

for idx, row in shap_df.iterrows():
    epoch = row['epoch']
    vals = row[feature_cols]
    
    # Sort features in non-increasing order of SHAP values
    sorted_vals = vals.sort_values(ascending=False)
    
    row_data = {'epoch': epoch}
    for rank, (feat, val) in enumerate(sorted_vals.items(), start=1):
        row_data[f'feature_{rank}'] = feat
        row_data[f'category_{rank}'] = get_attr_category(feat)
        row_data[f'shap_value_{rank}'] = val
        
    output_rows.append(row_data)

# 3. Create DataFrame and export to CSV
arranged_df = pd.DataFrame(output_rows)
output_filename = 'hidden_layer_7_shap_values_arranged_with_type.csv'
arranged_df.to_csv(output_filename, index=False)

print(f"Successfully generated and saved: {output_filename}")