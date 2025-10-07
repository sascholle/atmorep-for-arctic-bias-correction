import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import re

'''
source /work/ab1412/atmorep/pyenv/bin/activate
python /work/ab1412/atmorep/plotting/plot_losses.py

'''

  
output_id = 19413611

set_ylim=False
ylim_max=1.2

plot_window=False

''' Model Run Ids for Plotting

#### BERT vs forecast Sparsity Tests for Temperature #####
19304401 - bert, input masking of temp using 1,1,1,1
19305341 - bert, complete masked input field 0s, 50% target sparsity (and 1,1,1,1 as config)
19305366 - bert, 50% input 0s, 50% target (and 1,1,1,1 as config)
19305879 - bert, 50% input 0s, 50% target (and 0.5, 0.9, 0.2, 0.05 as config)
19305768 - NaNs dont work - bert, 50% input nans, 50% target (and 0.5, 0.9, 0.2, 0.05 as config)

19305731 - forecast, 50% input 0s, 50% target (and 0.5, 0.9, 0.2, 0.05 as config)
19305748 - forecast, 50% input 0s, 50% target (and 0.5, 0.9, 0.2, 0.05 as config)

#### Corrected t2m model for Arctic fine-tune training ####
    ## second iteration with better t2m learning curriculum 
19305085 - v2.1 Wandb run: atmorep-6mb1bcla-19305085 trained on j2l0sz9j 0.9 masking of t2m corrected
19309940 - v2.2 Wandb run: atmorep-qb7ksr0h-19309940 trained on 6mb1bcla 0.5 masking
19313842 - v2.3 Wandb run: atmorep-y1gpdgaa-19313842 trained on qb7ksr0h 0.5 masking
19337083 - v2.4 Wandb run: atmorep-u9vvriz7-19337083 trained on y1gpdgaa 0.5 masking
19366232 - v2.5 Wandb run: atmorep-iuy5bnth-19366232 trained on u9vvriz7 0.1 masking

 
    ## initial first runs
19017100 - t2m corrected run v1.2 0.5 masking
19062614 - loaded model 0: model-_id : ugqn2s9m v1.1 0.5 masking

#### Forecast and BERT Baseline search ####
    ## used in presentation
17850993 - BERT and plain mse (not ensemble), old prediction ratios
17720794 - BERT, MES, old prediction ratios (17767562 - BERT and global like 17720794)
17720895 - forecast, MES, old prediction ratios (17767305 - forecast and global like 17720895)
    ## new prediction ratios
17159353 - BERT, MES, new prediction ratios
17138477 - forecast, MES, new prediction ratios

#### t2m with temp cross attention and decaying masking #####
19295934 - v8.2 (unfortunate doubling up).. maybe good for plotting
19300146 - v8 
19285543 - v7
19272003 - v6
19210229 - v4 trained on wldctg77
19155880 - v3 trained on dhap4i9v
19133753 - v2 trained on uo2r80k8
19111065 - v1 trained on wc5e2i3t

#### t2m tests ####
19094477 - v9
19094345 - v1.2 t2m with temperature cross attention
19062136 - v8.3 normal t2m masking 
19062198 - v8.2 no t2m masking 0.001
19058384 - v8.1 much higher masking at 0.9
19031053 - v7
18941949 - v6
18909849 - v5 
18719337 - 50% masked with target set to t2m bert 
18717159 - 50% masked t2m bert
18685688 - completely masked input t2m bert, all others not. No target sparsity because forecast, but full input sparsity  
    ## finetuning starting
18677812 - and again t2m bert v4
18612021 - and again t2m bert v3, third time continued fine tuning on t2m using 18584657
18584657 - t2m bert v2, continued t2m fine tuning, not converged yet 
18543509 - t2m bert v1, really high t2m loss weighting, mse and stats - solid learning, not converged yet 
18524597 - t2m bert, sparsity input and target 90% - no learning 
18522685 - t2m bert, mid t2m weight - good learning - t2m has values 
18347853 - t2m bert 
    ## checking t2m full longitude range -90-90 - no difference geo selection nothing
18994864 - re-trained t2m model b9h8xdoz on full lat range
18994795 - training on wc5e2i3t again to see improvement 

    ## only MSE ensemble
17831587 - BERT and MSE ens, 480 samples and 10 hours to check training 
17828425 - BERT and MSE ens, 1024 samples per epoch - weird training plotting...
17824426 - BERT and MSE ens as the paper has

#### Forecasting Global only with precip target ####
17720340 - precip target forecast baseline global, 128 epochs 
17714187 - precip target forecast baseline global with updated config to match old models as best as possible (only prediction rations changed)
17698436 - precip target forecast baseline global no target or input masking, normal prediction ratios
17691529 - precip target forecast baseline arctic, no target or input masking, normal prediction ratios

#### Forecast Arctic only predicting and targeting precip ####
17555479 - 0's input double check
17546671 - 0's for input in precip
17398874 - 0%
17398873 - 10%
17413071 - 50%
17398877 - 90% 
17398871 - 100% - not possible
17398921 - 99.9%
17390826 - tailored MSE
17321944 - Forecast, MSE only, does grad loss and mse differ? why?
17307463 - Forecast, only predicting precip, Mse and stats, fully masked precip though this shouldnt change anything.
17307451 - Forecast, only predicting precip, stats and mse for Forecast arctic with minimal masking (hopefully same as above masking doesn't change anything)

#### BERT ERA #####
17254410 - Bert Arctic, target precip and no masking of other fields. 
17254035 - OOTB, old MSE 4th check
17191348 - OOTB, old MSE as well 3rd check 
17183688 - no arctic default 2nd check
17159066 - Bert default prediction and input masking, mse and stats, 70% target sparsity, But still sparsity setting - could be messing up precip
17138477 - Bert local, why no loss lines in plot??, mse ensemble and stats, 70% target sparsity, arctic, but default input masking and prediction ratios, 
17138473 - Bert local, mse ensemble only, 70% target sparsity, arctic, but default input masking and prediction ratios, 
17133443 - Bert local, only precip prediction and target, 70% target masking, default masking,  
17116108 - Bert and global temp 

#### Only predicting precip: ####
    ## Bert PLUS complete masking of precip too
    all seem to have similar total loss
    why is global temp here better for temp loss?
    does target sparsity really have no effect on BERT?
    MES seems to have less chaotic loss than just MSE ens
17086740 - Bert, mse ens, global temp, 70% target sparsity, should not have any change seeing as the sparsity shouldnt effect the BERT tokens - update: why would this be the case??, best loss out of three ~ 0.09
17116097 - Bert, mse ens, local temp, precip 70% target sparsity
16975985 - Bert, MES, local temp, no target sparsity, loss improved - BERT better than forecast, even with NO input?

    ## How does forecasting handle target sparsity so well? The grad loss is a bit over the place, but still total loss is lower than model *213 without target sparsity?? Mmh maybe thats because of the loss. Mse ens lower loss than with stats
    And why do the other fields respond by converging so well, especially temp, even though they are not being predicted or targeted?
    AND why does temp actually perform better with global norms ~0.01, vs local ~0.11??
17112711 - fore, 70% sparsity, mse ens, local temp, only predicting precip, 100% input sparsity - means nothing for forecast
17086747 - fore, 70% sparsity, global temp, only mse ens, forecast, precip check 2 - same peak around epoch 2-3
17019617 - fore, 70% sparsity, global temp, only mse ens, forecast, precip , global temp
    ## All three of these should be showing the same model: only predicting and thus targeting precip, local temp, MES, 
    Why does the loss of the other fields so chaotic, especially in comparison to the model with target sparsity? --- makes me think the target sparsity code needs checking... this surely shouldn't be the case
16953415 - fore, MES,  default targets, 100% input masking precip - non-sensicle for fore, loss improved
16951213 - fore, MES, double check, only predicting and target precip - higher loss though?
16441376 - fore, MES, only predicting and target precip, weird that this has better loss than model without only targeting precip?? - oh wait thats not weird. The predict list is adopted for target list so effectively its the same.  

#### Only predicting t2m: ####
16779906 - fore, t2m 50 epochs
16799165 - fore, t2m 500 epochs
'''

output_path = f"/work/ab1412/atmorep/output/output_{output_id}.txt"


def extract_config_parameters(file_path):
    """Extract key configuration parameters from the output file."""
    params = {}
    
    with open(file_path, 'r') as file:
        content = file.read()

        # Extract Wandb run information
        wandb_run_match = re.search(r'0: Wandb run: (.*?)$', content, re.MULTILINE)
        if wandb_run_match:
            params['wandb_run'] = wandb_run_match.group(1)
        
        # extract loaded model id
        loaded_model_match = re.search(r'0: Loaded model id = (\w+)', content)
        if loaded_model_match:
            params['loaded_model'] = loaded_model_match.group(1)

        # Extract Fields
        # Extract multi-line Fields (everything up to fields_prediction)
        fields_match = re.search(r'0: fields : \[(.*?)]\s*(?:0: fields_prediction|0: fields_targets|$)', content, re.DOTALL)
        if fields_match:
            params['fields'] = fields_match.group(1).strip()
        
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
        
        # # More detailed fields info 
        # fields_info_match = re.search(r'0: fields : (.*?)$', content, re.MULTILINE)
        # if fields_info_match:
        #     params['fields_info_detailed'] = fields_info_match.group(1)
            
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
        
        # add 0: self.lats : (71,)
        lats_match = re.search(r'0: self.lats : \((\d+),\)', content)
        if lats_match:
            params['lats_shape'] = int(lats_match.group(1))
            
        # add 0: self.lons : (1440,)
        lons_match = re.search(r'0: self.lons : \((\d+),\)', content)
        if lons_match:
            params['lons_shape'] = int(lons_match.group(1))

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
corrected_t2m = 'validation loss for corrected_t2m :'
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
corrected_t2m_values = []

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
        elif corrected_t2m in line:
            value = float(line.split(corrected_t2m)[1].strip())
            corrected_t2m_values.append(value)
        

# Convert lists to arrays
precip_values = np.array(precip_values)
total_loss_values = np.array(total_loss_values)
u_values = np.array(u_values)
v_values = np.array(v_values)
q_values = np.array(q_values)
z_values = np.array(z_values)
temp_values = np.array(temp_values)
t2m_values = np.array(t2m_values)
corrected_t2m_values = np.array(corrected_t2m_values)
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
variables = ["velocity_u", "velocity_v", "velocity_z", "temperature", "specific_humidity", "total_precip", "total_loss", "t2m", "corrected_t2m"]
data = {
    "velocity_u": u_values,
    "velocity_v": v_values,
    "velocity_z": z_values,
    "temperature": temp_values,
    "specific_humidity": q_values,
    "total_precip": precip_values,
    "total_loss": total_loss_values,
    "t2m": t2m_values,
    "corrected_t2m": corrected_t2m_values
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
ax.text(0.64, 0.98, mean_text, transform=ax.transAxes, fontsize=9,
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
if t2m_values.size > 0:  # Check if there are any T2M values
    ax.plot(epochs_val, t2m_values, label='T2M Loss', color='gray')
if corrected_t2m_values.size > 0:  # Check if there are any corrected T2M values
    ax.plot(epochs_val, corrected_t2m_values, label='Corrected T2M Loss', color='cyan') 

# Plot all training stats per batch (x = epoch + batch_idx/total_batches)
training_x = [e + (b-1)/5 for e, b in zip(training_loss_epochs, training_batch_indices)]  # adjust denominator if not 5 batches/epoch
ax.plot(training_x, training_loss_stddev, 'o-', label='Training Stddev', color='lightgray', markersize=1)
ax.plot(training_x, training_loss_values, marker='o', label='Training Grad Loss', color='black', linewidth=0.75, linestyle='--', markersize=1)
ax.plot(training_x, training_loss_mse, marker='o', label='Training MSE', color='gray', linewidth=0.75, linestyle='--', markersize=1)
if plot_window==True:
    window = 20  # Choose window size (number of points to average)
    if len(training_loss_values) >= window:
        moving_avg = np.convolve(training_loss_values, np.ones(window)/window, mode='valid')
        moving_x = training_x[:len(moving_avg)]
        ax.plot(moving_x, moving_avg, color='red', linewidth=2, label='Training Grad Loss Trend', alpha=0.7)

# Add labels and title
ax.set_xlabel('Epoch', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_ylabel('Loss', fontsize=18)
ax.set_title(f'AtmoRep Validation Losses over Epochs (Run ID: {output_id})', fontsize=23)
ax.legend(loc='upper right', fontsize=10)

if set_ylim:
    ax.set_ylim(0,ylim_max)

ax.xaxis.set_minor_locator(MultipleLocator(1))  # or 1 for epochs
ax.tick_params(axis='x', which='minor', length=5)  # adjust length as needed
ax.tick_params(axis='x', which='minor', labelbottom=False)  # hide minor tick labels

# Save the plot to a file
output_file = f"/work/ab1412/atmorep/plotting/losses_plot_id_{output_id}.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Plot saved as {output_file}")

