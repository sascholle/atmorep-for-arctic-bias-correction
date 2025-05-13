import matplotlib.pyplot as plt
import numpy as np
import re

output_id = 16779906
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

    return params


# Get configuration parameters
config_params = extract_config_parameters(output_path)

# Create the parameter text
param_text = "Model Configuration:\n"
for k, v in config_params.items():
    param_text += f"{k}: {v}\n"


# Load the data with all target phrases
total_loss = 'validation loss for strategy=forecast at epoch'
precip_loss = 'validation loss for total_precip :'
u_loss = 'validation loss for velocity_u :'
v_loss = 'validation loss for velocity_v :'
q_loss = 'validation loss for specific_humidity :'
z_loss = 'validation loss for velocity_z :'
temp_loss = 'validation loss for temperature :'
t2m = 'validation loss for t2m :'
training_loss_pattern = r'0: epoch: (\d+) \[4/5 \(80%\)\]\s+Loss: ([\d.]+) : ([\d.]+) :: ([\d.]+)'

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
training_loss_values = []
training_loss_mse = []
training_loss_stddev = []

with open(output_path, 'r') as file:
    content = file.read()

     # Extract training loss with regex
    training_matches = re.findall(training_loss_pattern, content)
    for match in training_matches:
        epoch = int(match[0])
        loss = float(match[1])
        mse = float(match[2])
        stddev = float(match[3])
        training_loss_epochs.append(epoch)
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

# Optional: Add grid in the back (behind the data)
ax.set_axisbelow(True)

# Add text box for hyperparameters
fig.text(0.93, 0.13, param_text, fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.8))

# Adjust layout to make room for the text box
#plt.subplots_adjust(right=0.75)

# Plot all variables
ax.plot(total_loss_values, label='Total Loss', color='red')
ax.plot(precip_values, label='Precipitation Loss', color='blue')
ax.plot(u_values, label='U Velocity Loss', color='green')
ax.plot(v_values, label='V Velocity Loss', color='orange')
ax.plot(z_values, label='Z Velocity Loss', color='brown')
ax.plot(q_values, label='Specific Humidity Loss', color='purple')
ax.plot(temp_values, label='Temperature Loss', color='pink')
ax.plot(t2m_values, label='T2M Loss', color='gray')


# Add training loss to the same plot
# We need to align the epochs since training_loss_epochs starts at 1
if len(training_loss_epochs) > 0 and len(training_loss_values) > 0:
    # Create epochs array that matches the validation loss x-axis
    aligned_epochs = np.arange(len(total_loss_values))
    # Plot only the epochs we have data for
    training_plot_indices = [i for i in aligned_epochs if i+1 in training_loss_epochs]
    training_values = [training_loss_values[list(training_loss_epochs).index(i+1)] for i in training_plot_indices]
    
    # Plot the training loss
    ax.plot(training_plot_indices, training_values, 'o-', 
            label='Training Loss: grad loss total', color='black', linewidth=2, markersize=4)


# Add labels and title
ax.set_xlabel('Epoch', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=18)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title(f'AtmoRep Validation Losses over Epochs (Run ID: {output_id})', fontsize=23)
ax.legend(loc='upper right', fontsize=12)

# Save the plot to a file
output_file = f"/work/ab1412/atmorep/plotting/losses_plot_id_{output_id}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as {output_file}")

'''
source /work/ab1412/atmorep/pyenv/bin/activate
module load python3
python /work/ab1412/atmorep/plotting/plot_losses.py

'''
