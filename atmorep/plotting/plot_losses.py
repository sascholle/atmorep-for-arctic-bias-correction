import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import re

'''
source /work/ab1412/atmorep/pyenv/bin/activate
python /work/ab1412/atmorep/plotting/plot_losses.py

'''

 
output_id =  17850993 

## Forecast and BERT Baseline search

# 17850993 - BERT and plan mse (not ensemble)
# 17831587 - BERT and only MSE, 480 samples and 10 hours to check training 
# 17828425 - BERT and only MSE, 1024 samples per epoch - weird training plotting...
# 17824426 - BERT and only MSE as the paper has

# 17767562 - BERT and global like 17720794
# 17767305 - forecast and global like 17720895

# 17720895 - forecast baseline global, 128 epochs 
# 17720794 - bert baseline global, 128 epochs
# 17720340 - precip target forecast baseline global, 128 epochs 
# 17714187 - precip target forecast baseline global with updated config to match old models as best as possible (only prediction rations changed)
# 17698436 - precip target forecast baseline global no target or input masking, normal prediction ratios
# 17691529 - precip target forecast baseline arctic, no target or input masking, normal prediction ratios

## Forecast Arctic
# 17555479 - 0's input double check 
# 17546671 - 0's for input in precip
# 17398874 - 0%
# 17398873 - 10%
# 17413071 - 50%
# 17398877 - 90% 
# 17398871 - 100% - not possible
# 17398921 - 99.9%
# 17390826 - tailored MSE
# 17321944 - Forecast, MSE only, does grad loss and mse differ? why?
# 17307463 - baseline Forecast, Arctic, Mse and stats, fully masked precip though this shouldnt change anything. 
# 17307451 - Baseline stats and mse for Forecast arctic with minimal masking (hopefully same as above masking doesn't change anything)

## BERT ERA
# 17254410 - Bert Arctic, target precip and no masking of other fields. 
# 17254035 - OOTB, old MSE 4th chech 
# 17191348 - OOTB, old MSE as well 3rd check 
# 17183688 - no arctic default 2nd check
# 17159353 - total default for Arctic, no sparsity setting
# 17159066 - total default, no Arctic? But still sparsity setting - could be messing up precip
# 17138477 - Bert local, arctic, default masking and prediction ratios MSE and Ensemble - Should be the absolute baseline for BERT for 2021
# 17138473 - Bert local, default masking and prediction ratios MSE only
# 17133443 - Bert local, default masking, precip prediction and target 
# 17116108 - Bert and global temp 
# 17116097 - Bert and local temp 
# 17112711 - Forecast and local temp

# 17086740 - Sparsity and BERT, should not have any change seeing as the sparsity shouldnt effect the BERT tokens, global temp
# 17086747 - 30% sparsity, forecast, precip check 2 - same peak around epoch 2-3, global temp
# 17019617 - 30% sparsity, forecast, precip , global temp
# 16975985 - BEEERTTTT local
# 16953415 - mask with 1's but in forecast 
# 16951213 - precip double check forecast 
# 16779906 - t2m 50 epochs
# 16799165 - t2m 500 epochs
# 16441376 - precip forecast loss 

output_path = f"/work/ab1412/atmorep/output/output_{output_id}.txt"


def extract_config_parameters(file_path):
    """Extract key configuration parameters from the output file."""
    params = {}
    
    with open(file_path, 'r') as file:
        content = file.read()
        
        # Extract Fields
        fields_match = re.search(r'0: Fields: \[(.*?)\]', content)
        if fields_match:
            params['Fields'] = fields_match.group(1)
        
        # Simple approach for fields_prediction - just get the whole line
        for line in content.split('\n'):
            if '0: fields_prediction :' in line:
                # Get everything after "0: fields_prediction :"
                value = line.split('0: fields_prediction :')[1].strip()
                params['fields_prediction'] = value
                break
        
        # Extract fields_targets
        targets_match = re.search(r'0: fields_targets : \[(.*?)\]', content)
        if targets_match:
            params['fields_targets'] = targets_match.group(1)
        
        # More detailed fields info 
        fields_info_match = re.search(r'0: fields : (.*?)$', content, re.MULTILINE)
        if fields_info_match:
            params['fields_info_detailed'] = fields_info_match.group(1)
            
        # Extract batch size and epochs
        batch_match = re.search(r'0: batch_size : (\d+)', content)
        if batch_match:
            params['batch_size'] = batch_match.group(1)
            
        epochs_match = re.search(r'0: num_epochs : (\d+)', content)
        if epochs_match:
            params['num_epochs'] = epochs_match.group(1)
            
        # Extract samples per epoch
        samples_match = re.search(r'0: num_samples_per_epoch : (\d+)', content)
        if samples_match:
            params['samples_per_epoch'] = samples_match.group(1)
            
        # Extract validation samples
        val_samples_match = re.search(r'0: num_samples_validate : (\d+)', content)
        if val_samples_match:
            params['num_samples_validate'] = val_samples_match.group(1)
            
        # Extract validation batch size
        val_batch_match = re.search(r'0: batch_size_validation : (\d+)', content)
        if val_batch_match:
            params['batch_size_validation'] = val_batch_match.group(1)
    
        # Extract training years range
        years_train_match = re.search(r'0: years_train : (.*?)$', content, re.MULTILINE)
        if years_train_match:
            params['years_train'] = years_train_match.group(1).strip()
        
        # Extract validation years range
        years_val_match = re.search(r'0: years_val : (.*?)$', content, re.MULTILINE)
        if years_val_match:
            params['years_val'] = years_val_match.group(1).strip()
        
        # Add loss function
        losses_match = re.search(r'0: losses : (.*?)$', content, re.MULTILINE)
        if losses_match:
            params['losses'] = losses_match.group(1).strip()
        
        # Add BERT strategy
        bert_strategy_match = re.search(r'0: BERT_strategy : (.*?)$', content, re.MULTILINE)
        if bert_strategy_match:
            params['BERT_strategy'] = bert_strategy_match.group(1).strip()
        
        # Add forecast_num_tokens
        forecast_tokens_match = re.search(r'0: forecast_num_tokens : (.*?)$', content, re.MULTILINE)
        if forecast_tokens_match:
            params['forecast_num_tokens'] = forecast_tokens_match.group(1).strip()
        
        # add geo_range_sampling to params 
        geo_range_match = re.search(r'0: geo_range_sampling : (.*?)$', content, re.MULTILINE)
        if geo_range_match:
            params['geo_range_sampling'] = geo_range_match.group(1).strip()

        # add if sparse_target is set to true or false
        sparse_target_match = re.search(r'0: sparse_target : (.*?)$', content, re.MULTILINE)
        if sparse_target_match:
            params['sparse_target'] = sparse_target_match.group(1).strip()

        # add sparse_target_field
        sparse_target_match = re.search(r'0: sparse_target_field : (.*?)$', content, re.MULTILINE)
        if sparse_target_match:
            params['sparse_target_field'] = sparse_target_match.group(1).strip()
        
        # add sparse_target_sparsity
        sparse_sparsity_match = re.search(r'0: sparse_target_sparsity : (.*?)$', content, re.MULTILINE)
        if sparse_sparsity_match:
            params['sparse_target_sparsity'] = sparse_sparsity_match.group(1).strip()

        # add mask input field 
        mask_input_field_match = re.search(r'0: mask_input_field : (.*?)$', content, re.MULTILINE)
        if mask_input_field_match:  
            params['mask_input_field'] = mask_input_field_match.group(1).strip()

        # add mask input value
        mask_input_value_match = re.search(r'0: mask_input_value : (.*?)$', content, re.MULTILINE)
        if mask_input_value_match:
            params['mask_input_value'] = mask_input_value_match.group(1).strip()            
        
    return params

# Create a formatted parameter text with line breaks for readability
def format_param_text(params, max_line_length=110):
    formatted_text = "Model Configuration:\n"
    
    for k, v in params.items():
        # For other long values, truncate or wrap them
        if isinstance(v, str) and len(v) > max_line_length:
            if k in ['years_train', 'years_val', 'Fields']:
                formatted_text += f"{k}: {v[:max_line_length]}...\n"
            else:
                # Try to wrap the text
                words = v.split()
                line = k + ": "
                for word in words:
                    if len(line + word) > max_line_length:
                        formatted_text += line + "\n  "
                        line = word + " "
                    else:
                        line += word + " "
                formatted_text += line + "\n"
        else:
            formatted_text += f"{k}: {v}\n"
         
    return formatted_text

# Get configuration parameters
config_params = extract_config_parameters(output_path)

# Create the formatted parameter text
param_text = format_param_text(config_params)

# Load the data with all target phrases
total_loss = 'validation loss for strategy='
precip_loss = 'validation loss for total_precip :'
u_loss = 'validation loss for velocity_u :'
v_loss = 'validation loss for velocity_v :'
q_loss = 'validation loss for specific_humidity :'
z_loss = 'validation loss for velocity_z :'
temp_loss = 'validation loss for temperature :'
t2m = 'validation loss for t2m :'
training_loss_pattern = r'0: epoch: (\d+) \[(\d+)/(\d+) \((\d+)%\)\]\s+Loss: ([\d.]+) : ([\d.]+) :: ([\d.]+)'

# Initialize lists for all variables
total_loss_values = []
precip_values = []
u_values = []
v_values = []
q_values = []
z_values = []
temp_values = []
t2m_values = []

training_loss_epochs = []
training_batch_indices = []
training_loss_values = []
training_loss_mse = []
training_loss_stddev = []

with open(output_path, 'r') as file:
    content = file.read()

     # Extract training loss with regex
    training_matches = re.findall(training_loss_pattern, content)
    for match in training_matches:
        epoch = int(match[0])
        batch_idx = int(match[1])
        loss = float(match[4])
        mse = float(match[5])
        stddev = float(match[6])
        training_loss_epochs.append(epoch)
        training_batch_indices.append(batch_idx)
        training_loss_values.append(loss)
        training_loss_mse.append(mse)
        training_loss_stddev.append(stddev)

    for line in content.split('\n'):
        if precip_loss in line:
            value = float(line.split(precip_loss)[1].strip())
            precip_values.append(value)
        elif total_loss in line:
            value = float(line.split(':')[-1].strip())
            total_loss_values.append(value)
        elif u_loss in line:
            value = float(line.split(u_loss)[1].strip())
            u_values.append(value)
        elif v_loss in line:
            value = float(line.split(v_loss)[1].strip())
            v_values.append(value)
        elif q_loss in line:
            value = float(line.split(q_loss)[1].strip())
            q_values.append(value)
        elif z_loss in line:
            value = float(line.split(z_loss)[1].strip())
            z_values.append(value)
        elif temp_loss in line:
            value = float(line.split(temp_loss)[1].strip())
            temp_values.append(value)
        elif t2m in line:
            value = float(line.split(t2m)[1].strip())
            t2m_values.append(value)
        

# Convert lists to arrays
precip_values = np.array(precip_values)
total_loss_values = np.array(total_loss_values)
u_values = np.array(u_values)
v_values = np.array(v_values)
q_values = np.array(q_values)
z_values = np.array(z_values)
temp_values = np.array(temp_values)
t2m_values = np.array(t2m_values)
training_loss_epochs = np.array(training_loss_epochs)
training_loss_values = np.array(training_loss_values)
training_loss_mse = np.array(training_loss_mse)
training_loss_stddev = np.array(training_loss_stddev)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Add grid with darker grey lines
ax.grid(True, linestyle='--', alpha=0.7, color='#a0a0a0')  # Darker grey grid lines
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.grid(which='major', axis='x', linestyle='--', alpha=0.7, color='#a0a0a0')

# Optional: Add grid in the back (behind the data)
ax.set_axisbelow(True)

# Add text box for hyperparameters
fig.text(0.93, 0.13, param_text, fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.8))


# Calculate mean values of variables (excluding first 10 values if requested)
# Add this section
means = {}
variables = ["velocity_u", "velocity_v", "velocity_z", "temperature", "specific_humidity", "total_precip", "total_loss"]
data = {
    "velocity_u": u_values,
    "velocity_v": v_values,
    "velocity_z": z_values,
    "temperature": temp_values,
    "specific_humidity": q_values,
    "total_precip": precip_values,
    "total_loss": total_loss_values
}

# Calculate means
for var_name in variables:
    if var_name in data:
        means[var_name] = np.mean(data[var_name])

# Create mean values text
mean_text = "Mean Values:\n"
for var, val in means.items():
    mean_text += f"{var}: {val:.4f}\n"

# Add text box with mean values
props = dict(boxstyle='round', facecolor='white', alpha=0.7)
ax.text(0.02, 0.98, mean_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)


# Adjust layout to make room for the text box
#plt.subplots_adjust(right=0.75)

# Plot all variables
epochs_val = np.arange(1, len(total_loss_values)+1)
ax.plot(epochs_val, total_loss_values, label='Total Loss', color='red')
ax.plot(epochs_val, precip_values, label='Precipitation Loss', color='blue')
ax.plot(epochs_val, u_values, label='U Velocity Loss', color='green')
ax.plot(epochs_val, v_values, label='V Velocity Loss', color='orange')
ax.plot(epochs_val, z_values, label='Z Velocity Loss', color='brown')
ax.plot(epochs_val, q_values, label='Specific Humidity Loss', color='purple')
ax.plot(epochs_val, temp_values, label='Temperature Loss', color='pink')
#ax.plot(epochs_val, t2m_values, label='T2M Loss', color='gray')

# Plot all training stats per batch (x = epoch + batch_idx/total_batches)
training_x = [e + (b-1)/5 for e, b in zip(training_loss_epochs, training_batch_indices)]  # adjust denominator if not 5 batches/epoch
ax.plot(training_x, training_loss_values, marker='o', label='Training Grad Loss', color='black', linewidth=1,linestyle='--', markersize=2)
ax.plot(training_x, training_loss_mse, marker='o', label='Training MSE', color='gray', linewidth=1, linestyle='--', markersize=2)
ax.plot(training_x, training_loss_stddev, 'o-', label='Training Stddev', color='lightgray', markersize=3)

# Add labels and title
ax.set_xlabel('Epoch', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title(f'AtmoRep Validation Losses over Epochs (Run ID: {output_id})', fontsize=23)
ax.legend(loc='upper right', fontsize=12)



#ax.set_ylim(0,1)


ax.xaxis.set_minor_locator(MultipleLocator(1))  # or 1 for epochs
ax.tick_params(axis='x', which='minor', length=5)  # adjust length as needed
ax.tick_params(axis='x', which='minor', labelbottom=False)  # hide minor tick labels

# Save the plot to a file
output_file = f"/work/ab1412/atmorep/plotting/losses_plot_id_{output_id}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as {output_file}")

