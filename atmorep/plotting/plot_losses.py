import matplotlib.pyplot as plt
import numpy as np
import re

output_id = 16449809
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
    
    return params


# Get configuration parameters
config_params = extract_config_parameters(output_path)

# Create the parameter text
param_text = "Model Configuration:\n"
for k, v in config_params.items():
    param_text += f"{k}: {v}\n"


# Load the data with all target phrases
precip_loss = 'validation loss for total_precip :'
total_loss = 'validation loss for strategy=forecast at epoch'
u_loss = 'validation loss for velocity_u :'
v_loss = 'validation loss for velocity_v :'
q_loss = 'validation loss for specific_humidity :'
z_loss = 'validation loss for velocity_z :'
temp_loss = 'validation loss for temperature :'

# Initialize lists for all variables
precip_values = []
total_loss_values = []
u_values = []
v_values = []
q_values = []
z_values = []
temp_values = []

with open(output_path, 'r') as file:
    for line in file:
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

# Convert lists to arrays
precip_values = np.array(precip_values)
total_loss_values = np.array(total_loss_values)
u_values = np.array(u_values)
v_values = np.array(v_values)
q_values = np.array(q_values)
z_values = np.array(z_values)
temp_values = np.array(temp_values)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Add text box for hyperparameters
fig.text(0.93, 0.13, param_text, fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.8))

# Adjust layout to make room for the text box
#plt.subplots_adjust(right=0.75)

# Plot all variables
ax.plot(precip_values, label='Precipitation Loss', color='blue')
ax.plot(total_loss_values, label='Total Loss', color='orange')
ax.plot(u_values, label='U Velocity Loss', color='green')
ax.plot(v_values, label='V Velocity Loss', color='red')
ax.plot(q_values, label='Specific Humidity Loss', color='purple')
ax.plot(z_values, label='Z Velocity Loss', color='brown')
ax.plot(temp_values, label='Temperature Loss', color='pink')

# Add labels and title
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title(f'AtmoRep Validation Losses over Epochs (Run ID: {output_id})')
ax.legend()

# Save the plot to a file
output_file = f"losses_plot_id_{output_id}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as {output_file}")
