import re
import os

# filepath: /work/ab1412/atmorep/plotting/plot_losses.py
# --- Step 1: Extract all model run IDs from the section ---
def extract_model_ids_from_script(script_path):
    with open(script_path, 'r') as f:
        content = f.read()
    # Find the section
    section_match = re.search(r"''' Model Run Ids for Plotting(.*?)'''", content, re.DOTALL)
    if not section_match:
        print("Could not find model run IDs section.")
        return []
    section = section_match.group(1)
    # Find all numbers with at least 8 digits (your IDs)
    ids = re.findall(r'\b\d{8,}\b', section)
    return sorted(set(ids))

# --- Step 2: Extract config parameters from output file ---
def extract_config_parameters(file_path):
    params = {}
    if not os.path.exists(file_path):
        return params
    with open(file_path, 'r') as file:
        content = file.read()
        wandb_run_match = re.search(r'0: Wandb run: (.*?)$', content, re.MULTILINE)
        if wandb_run_match:
            params['wandb_run'] = wandb_run_match.group(1)
        loaded_model_match = re.search(r'0: Loaded model id = (\w+)', content)
        if loaded_model_match:
            params['loaded_model'] = loaded_model_match.group(1)
        batch_match = re.search(r'0: batch_size : (\d+)', content)
        if batch_match:
            params['batch_size'] = batch_match.group(1)
        epochs_match = re.search(r'0: num_epochs : (\d+)', content)
        if epochs_match:
            params['num_epochs'] = epochs_match.group(1)
        losses_match = re.search(r'0: losses : (.*?)$', content, re.MULTILINE)
        if losses_match:
            params['losses'] = losses_match.group(1).strip()
        bert_strategy_match = re.search(r'0: BERT_strategy : (.*?)$', content, re.MULTILINE)
        if bert_strategy_match:
            params['BERT_strategy'] = bert_strategy_match.group(1).strip()
        forecast_tokens_match = re.search(r'0: forecast_num_tokens : (.*?)$', content, re.MULTILINE)
        if forecast_tokens_match:
            params['forecast_num_tokens'] = forecast_tokens_match.group(1).strip()
        geo_range_match = re.search(r'0: geo_range_sampling : (.*?)$', content, re.MULTILINE)
        if geo_range_match:
            params['geo_range_sampling'] = geo_range_match.group(1).strip()
        years_train_match = re.search(r'0: years_train : (.*?)$', content, re.MULTILINE)
        if years_train_match:
            params['years_train'] = years_train_match.group(1).strip()
        years_val_match = re.search(r'0: years_val : (.*?)$', content, re.MULTILINE)
        if years_val_match:
            params['years_val'] = years_val_match.group(1).strip()
    return params

# --- Step 3: Build Markdown table ---
def build_markdown_table(model_ids, config_dicts, fields_to_show):
    header = "| Run ID | " + " | ".join(fields_to_show) + " |\n"
    header += "|--------|" + "|".join(["---"]*len(fields_to_show)) + "|\n"
    rows = []
    for run_id in model_ids:
        params = config_dicts.get(run_id, {})
        row = f"| {run_id} | " + " | ".join([params.get(f, "") for f in fields_to_show]) + " |"
        rows.append(row)
    return header + "\n".join(rows)

# --- Main script ---
if __name__ == "__main__":
    script_path = "/work/ab1412/atmorep/plotting/plot_losses.py"
    output_dir = "/work/ab1412/atmorep/output"
    fields_to_show = [
        "wandb_run", "loaded_model", "batch_size", "num_epochs", "losses",
        "BERT_strategy", "years_train", "years_val"
    ]
    model_ids = extract_model_ids_from_script(script_path)
    config_dicts = {}
    for run_id in model_ids:
        output_path = os.path.join(output_dir, f"output_{run_id}.txt")
        config_dicts[run_id] = extract_config_parameters(output_path)
    markdown_table = build_markdown_table(model_ids, config_dicts, fields_to_show)
    # Save to markdown file
    with open("/work/ab1412/atmorep/plotting/model_runs_overview.md", "w") as f:
        f.write(markdown_table)
    print("Markdown table saved as model_runs_overview.md")