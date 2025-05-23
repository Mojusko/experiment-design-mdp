import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import json
import re # Import re for pattern matching
import yaml # Added for loading config files

def parse_filename(filename):
    base = os.path.basename(filename)
    model_name_from_file = None

    # General pattern to try and extract a model name suffix first
    # Model names are typically alphanumeric with underscores, not containing '-' used in core IDs.
    # This regex tries to find a model name like "-modelname.json" at the end.
    # It assumes the model name is the last segment before .json if that segment doesn't look like a seed.
    # Example: metrics-core_id-seed-modelname.json or metrics-core_id-modelname.json
    
    core_id_part = base.replace("metrics-", "").replace(".json", "")
    
    # Try to split off a model name if it's the last part and doesn't look like a typical seed number
    # This is a heuristic. A more robust way is if model names have a very distinct pattern or prefix.
    # Current per-model saving: <original_exp_id>-<model_name>
    # So, if original_exp_id is "foo-123", per-model is "foo-123-sunny".
    
    # Let's try to match the longest possible core ID first, then see if a known model name follows.
    # This is hard to do generically without knowing all model names.
    # A simpler approach: make model name optional in all specific regexes.

    # Regex for optional model name suffix: (?:-([a-zA-Z0-9_]+))?
    # This captures an optional group starting with '-' followed by alphanumeric/underscore.
    opt_model_suffix_re = r"(?:-([a-zA-Z0-9_]+))?"

    # Check for other experiment types first.
    # V-comparison files typically don't have model suffixes in the same way.
    if "withV" in base or "noV" in base: # Assuming these don't have model suffixes
        v_type = "With V" if "withV" in base else "No V"
        return ("v_comparison", v_type, None)

    # Match single lambda format.
    # e.g., metrics-lambda-dsn-mult-mul-lambda15-6.json or metrics-mul-lambda1-3.json
    # New: metrics-mul-lambda1-3-sunny.json
    if (match := re.search(r"lambda([\d.]+)-(\d+)" + opt_model_suffix_re + r"\.json$", base)):
        lambda_val = float(match.group(1))
        # seed = int(match.group(2))
        model_name_from_file = match.group(3) # Will be None if not present
        return ("lambda", lambda_val, model_name_from_file)
    # Match old single lambda format where "lambda" isn't in the value part:
    # e.g. metrics-mul-0.1-1.json or metrics-num-0.1-1.json
    # New: metrics-mul-0.1-1-sunny.json
    elif (match := re.search(r"(mul|num)-([\d.]+)-(\d+)" + opt_model_suffix_re + r"\.json$", base)):
        # Group 1 is mul/num, Group 2 is lambda_val, Group 3 is seed
        lambda_val = float(match.group(2))
        model_name_from_file = match.group(4) # Optional model name
        return ("lambda", lambda_val, model_name_from_file)
    # Match lambda_sparsity format: metrics-muls-lambda_sparsity0.01-1.json
    # New: metrics-muls-lambda_sparsity0.01-1-sunny.json
    elif (match := re.search(r"lambda_sparsity([\d.]+)-(\d+)" + opt_model_suffix_re + r"\.json$", base)):
        lambda_sparsity_val = float(match.group(1))
        # seed = int(match.group(2))
        model_name_from_file = match.group(3)
        return ("lambda_sparsity", lambda_sparsity_val, model_name_from_file)
    # Match design frequency filenames like metrics-design-freq-dsn-mult-ep25-df10-1.json
    # New: metrics-design-freq-dsn-mult-ep25-df10-1-sunny.json
    elif (match := re.search(r"ep(\d+)-df(\d+)-(\d+)" + opt_model_suffix_re + r"\.json$", base)):
        episodes = int(match.group(1))
        frequency = int(match.group(2))
        # seed = int(match.group(3)) # Seed not used for grouping key
        model_name_from_file = match.group(4)
        return ("design_frequency", (episodes, frequency), model_name_from_file)
    # Match rounds filenames: metrics-rounds-finding-15-1.json
    # New: metrics-rounds-finding-15-1-sunny.json
    elif (match := re.search(r"rounds-finding-(\d+)-(\d+)" + opt_model_suffix_re + r"\.json$", base)): # Adjusted to match example
        rounds_val = int(match.group(1))
        # seed = int(match.group(2))
        model_name_from_file = match.group(3)
        return ("rounds", rounds_val, model_name_from_file)
    else:
        # Match feedback filenames:
        # New format: metrics-feedback-dsn-mult-ep50-1.json
        # New format with model: metrics-feedback-dsn-mult-ep50-1-sunny.json
        # Old format: metrics-feedback-dsn-mult-1.json
        # Old format with model: metrics-feedback-dsn-mult-1-sunny.json
        
        # Pattern for new format (with episodes)
        feedback_pattern_ep = re.compile(r"metrics-feedback-(\w+)-(\w+)-ep(\d+)-(\d+)" + opt_model_suffix_re + r"\.json$")
        # Pattern for old format (without episodes explicitly)
        feedback_pattern_old = re.compile(r"metrics-feedback-(\w+)-(\w+)-(\d+)" + opt_model_suffix_re + r"\.json$")

        match_ep = feedback_pattern_ep.search(base)
        if match_ep:
            alg_code = match_ep.group(1)
            feed_code = match_ep.group(2)
            episodes = int(match_ep.group(3))
            # seed = int(match_ep.group(4))
            model_name_from_file = match_ep.group(5)
            return ("feedback", (alg_code, feed_code, episodes), model_name_from_file)
        
        match_old = feedback_pattern_old.search(base)
        if match_old:
            # Ensure it's not an "ep" style by checking parts
            parts_check = base.replace("metrics-feedback-", "").replace(".json", "").split('-')
            # Check if "ep" is in any part before the seed and optional model name.
            # If model name is present, parts_check could be [alg, feed, seed, model] or [alg, feed, model_if_seed_is_absent_in_name]
            # This check needs to be careful.
            # A simpler heuristic: if the "episodes" part (group 3 of feedback_pattern_old) is not "ep<number>", it's old.
            # The regex for old format already ensures group(3) is (\d+), so it won't match "ep50".
            alg_code = match_old.group(1)
            feed_code = match_old.group(2)
            # seed = int(match_old.group(3))
            model_name_from_file = match_old.group(4)
            return ("feedback", (alg_code, feed_code, -1), model_name_from_file) # -1 indicates legacy/no episode info

        # Fallback for other unrecognized patterns or if the above are too strict
        print(f"Warning: Unrecognized filename pattern for {base}. Skipping or using fallback.")
        return ("unknown", base, None)


def safe_load_data(filename):
    # Try loading as plain text numeric data.
    try:
        data = np.loadtxt(filename)
        if data.size > 0:
            return data
    except Exception:
        pass

    # Try to load as a JSON dump.
    try:
        with open(filename, 'r') as f:
            loaded_json = json.load(f)
        
        # If the loaded JSON is a dictionary and contains typical metric keys, return it as is.
        if isinstance(loaded_json, dict) and \
           ("preference_error" in loaded_json or "cosine_error" in loaded_json):
            return loaded_json # Return the dictionary
            
        # If it's a dictionary but doesn't have the specific metric keys,
        # try to extract a single numeric value (original fallback behavior).
        elif isinstance(loaded_json, dict):
            for value in loaded_json.values():
                if isinstance(value, (int, float)):
                    return value # Return the first numeric value found
        
        # If the loaded JSON is directly a number (e.g. file contains just "0.5")
        elif isinstance(loaded_json, (int, float)):
            return loaded_json

    except Exception: # Handles json.load errors or other issues
        pass

    return None # If all attempts fail

def plot_results_with_type(results, plot_type, model_name_filter=None, save_plots=False, output_dir="."): # Added save_plots, output_dir
    # This function handles non-feedback experiments with numeric data.
    if not results:
        return

    keys = sorted(results.keys())
    valid_data = {k: [v for v in results[k] if v is not None] for k in keys}
    valid_keys = [k for k in keys if valid_data[k]]

    if not valid_keys:
        return

    # Check if data is numeric or dictionary-based
    if all(isinstance(v, (int, float)) for k in valid_keys for v in valid_data[k]):
        # Sort keys numerically
        valid_keys.sort()
        x_labels = [str(x) for x in valid_keys]

        means = [np.mean(valid_data[k]) for k in valid_keys]
        # Calculate Standard Error of the Mean (SEM = std / sqrt(n))
        sems = [(np.std(valid_data[k]) / np.sqrt(len(valid_data[k]))) if len(valid_data[k]) > 1 else 0 
                for k in valid_keys]

        plt.figure(figsize=(10, 6))
        plt.bar(x_labels, means, yerr=sems, capsize=5) # Use sems for yerr

        if plot_type == "lambda":
            plt.xlabel("Lambda Value")
        elif plot_type == "lambda_sparsity": # Added case for lambda_sparsity
            plt.xlabel("Lambda Sparsity Value")
        elif plot_type == "frequency":
            plt.xlabel("Estimation Frequency")
        elif plot_type == "rounds":
            plt.xlabel("Number of Rounds")
        elif plot_type == "v_comparison":
            plt.xlabel("Design Matrix Type")
        # Note: design_frequency is handled by plot_design_frequency_results now
        else:
            plt.xlabel(plot_type)

        plt.ylabel("Error")
        title = f"{plot_type.replace('_', ' ').title()} vs Value"
        if model_name_filter:
            title += f" (Model: {model_name_filter})"
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_plots:
            filename = os.path.join(output_dir, f"plot_{plot_type}{'_' + model_name_filter if model_name_filter else ''}.png")
            plt.savefig(filename)
            print(f"Saved plot to {filename}")
            plt.close()
    else:
        # Handle dictionary data (like v_comparison with multiple metrics)
        plot_comparison_results(valid_data, plot_type, model_name_filter=model_name_filter, save_plots=save_plots, output_dir=output_dir)

def plot_comparison_results(results, plot_type, model_name_filter=None, save_plots=False, output_dir="."): # Added save_plots, output_dir
    # For experiments with multiple metrics (e.g., preference_error, cosine_error)
    labels = list(results.keys())
    
    preference_means = []
    preference_sems = []
    cosine_means = []
    cosine_sems = []

    for alg in labels:
        pref_errors = [d["preference_error"] for d in results[alg] if isinstance(d, dict) and "preference_error" in d]
        cos_errors = [d["cosine_error"] for d in results[alg] if isinstance(d, dict) and "cosine_error" in d]
        
        n_pref = len(pref_errors)
        n_cos = len(cos_errors)
        
        preference_means.append(np.mean(pref_errors) if n_pref > 0 else 0)
        preference_sems.append((np.std(pref_errors) / np.sqrt(n_pref)) if n_pref > 1 else 0)
        
        cosine_means.append(np.mean(cos_errors) if n_cos > 0 else 0)
        cosine_sems.append((np.std(cos_errors) / np.sqrt(n_cos)) if n_cos > 1 else 0)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, preference_means, width, yerr=preference_sems, capsize=5, label="Preference Error") # Use sems
    plt.bar(x + width/2, cosine_means, width, yerr=cosine_sems, capsize=5, label="Cosine Error") # Use sems

    plt.xlabel("Design Matrix Type" if plot_type == "v_comparison" else plot_type.capitalize())
    plt.ylabel("Error")
    title = f"{plot_type.capitalize()} Comparison on Error Metrics"
    if model_name_filter:
        title += f" (Model: {model_name_filter})"
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    if save_plots:
        filename = os.path.join(output_dir, f"plot_{plot_type}_comparison{'_' + model_name_filter if model_name_filter else ''}.png")
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()

def plot_design_frequency_results(design_freq_results, model_name_filter=None, save_plots=False, output_dir="."): # Added save_plots, output_dir
    # Create a grouped bar chart comparing both metrics for each (episodes, frequency) pair.
    # Sort keys first by episodes, then by frequency for consistent plotting order
    labels = sorted(design_freq_results.keys(), key=lambda x: (x[0], x[1]))
    
    # Prepare data, ensuring we handle cases where a metric might be missing for a run
    preference_means = []
    preference_sems = [] # Changed from stds to sems
    cosine_means = []
    cosine_sems = [] # Changed from stds to sems

    for key in labels:
        pref_errors = [d["preference_error"] for d in design_freq_results[key] if isinstance(d, dict) and "preference_error" in d]
        cos_errors = [d["cosine_error"] for d in design_freq_results[key] if isinstance(d, dict) and "cosine_error" in d]
        
        n_pref = len(pref_errors)
        n_cos = len(cos_errors)
        
        preference_means.append(np.mean(pref_errors) if n_pref > 0 else 0)
        preference_sems.append((np.std(pref_errors) / np.sqrt(n_pref)) if n_pref > 1 else 0) # Calculate SEM
        cosine_means.append(np.mean(cos_errors) if n_cos > 0 else 0)
        cosine_sems.append((np.std(cos_errors) / np.sqrt(n_cos)) if n_cos > 1 else 0) # Calculate SEM

    x = np.arange(len(labels))
    width = 0.35
    
    # Format labels for the x-axis
    x_labels = [f"Ep{ep}-Df{df}" for ep, df in labels]

    plt.figure(figsize=(12, 7)) # Adjusted size for potentially more labels
    plt.bar(x - width/2, preference_means, width, yerr=preference_sems, capsize=5, label="Preference Error") # Use sems
    plt.bar(x + width/2, cosine_means, width, yerr=cosine_sems, capsize=5, label="Cosine Error") # Use sems

    plt.xlabel("Configuration (Episodes - Design Frequency)")
    plt.ylabel("Error")
    title = "Design Frequency Experiment Results"
    if model_name_filter:
        title += f" (Model: {model_name_filter})"
    plt.title(title)
    plt.xticks(x, x_labels, rotation=45, ha="right") # Rotate labels for better readability
    plt.legend()
    plt.tight_layout()
    if save_plots:
        filename = os.path.join(output_dir, f"plot_design_frequency{'_' + model_name_filter if model_name_filter else ''}.png")
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()


def plot_lambda_dsn_est_results(lambda_results):
    # Create a single grouped bar chart comparing metrics for each (dsn, est) pair.
    
    # Sort keys first by dsn, then by est
    sorted_keys = sorted(lambda_results.keys(), key=lambda x: (x[0], x[1]))
    
    labels = [f"Dsn={dsn}, Est={est}" for dsn, est in sorted_keys]
    
    preference_means = []
    preference_sems = [] # Changed from stds to sems
    cosine_means = []
    cosine_sems = [] # Changed from stds to sems

    for key in sorted_keys:
        data_list = lambda_results[key]
        pref_errors = [d["preference_error"] for d in data_list if isinstance(d, dict) and "preference_error" in d]
        cos_errors = [d["cosine_error"] for d in data_list if isinstance(d, dict) and "cosine_error" in d]
        
        n_pref = len(pref_errors)
        n_cos = len(cos_errors)
        
        preference_means.append(np.mean(pref_errors) if n_pref > 0 else 0)
        preference_sems.append((np.std(pref_errors) / np.sqrt(n_pref)) if n_pref > 1 else 0) # Calculate SEM
        cosine_means.append(np.mean(cos_errors) if n_cos > 0 else 0)
        cosine_sems.append((np.std(cos_errors) / np.sqrt(n_cos)) if n_cos > 1 else 0) # Calculate SEM

    x = np.arange(len(labels))
    width = 0.35

    # Adjust figure size based on the number of bars
    fig_width = max(12, len(labels) * 0.8) # Ensure minimum width, scale with number of labels
    plt.figure(figsize=(fig_width, 7)) 
    
    plt.bar(x - width/2, preference_means, width, yerr=preference_sems, capsize=5, label="Preference Error") # Use sems
    plt.bar(x + width/2, cosine_means, width, yerr=cosine_sems, capsize=5, label="Cosine Error") # Use sems

    plt.xlabel("Lambda Configuration (Design, Estimation)")
    plt.ylabel("Error")
    plt.xticks(x, labels, rotation=45, ha="right") # Rotate labels for better readability
    plt.title("Error vs Lambda Configuration (Design & Estimation)")
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels overlapping

# plot_lambda_dsn_est_results was removed, so no changes needed there.

# This function is no longer needed as lambda_dsn_est experiment type is removed.
# def plot_lambda_dsn_est_results(lambda_results):
#     # Create a single grouped bar chart comparing metrics for each (dsn, est) pair.
#     
#     # Sort keys first by dsn, then by est
#     sorted_keys = sorted(lambda_results.keys(), key=lambda x: (x[0], x[1]))
#     
#     labels = [f"Dsn={dsn}, Est={est}" for dsn, est in sorted_keys]
#     
#     preference_means = []
#     preference_sems = [] # Changed from stds to sems
#     cosine_means = []
#     cosine_sems = [] # Changed from stds to sems

#     for key in sorted_keys:
#         data_list = lambda_results[key]
#         pref_errors = [d["preference_error"] for d in data_list if isinstance(d, dict) and "preference_error" in d]
#         cos_errors = [d["cosine_error"] for d in data_list if isinstance(d, dict) and "cosine_error" in d]
#         
#         n_pref = len(pref_errors)
#         n_cos = len(cos_errors)
#         
#         preference_means.append(np.mean(pref_errors) if n_pref > 0 else 0)
#         preference_sems.append((np.std(pref_errors) / np.sqrt(n_pref)) if n_pref > 1 else 0) # Calculate SEM
#         cosine_means.append(np.mean(cos_errors) if n_cos > 0 else 0)
#         cosine_sems.append((np.std(cos_errors) / np.sqrt(n_cos)) if n_cos > 1 else 0) # Calculate SEM

#     x = np.arange(len(labels))
#     width = 0.35

#     # Adjust figure size based on the number of bars
#     fig_width = max(12, len(labels) * 0.8) # Ensure minimum width, scale with number of labels
#     plt.figure(figsize=(fig_width, 7)) 
#     
#     plt.bar(x - width/2, preference_means, width, yerr=preference_sems, capsize=5, label="Preference Error") # Use sems
#     plt.bar(x + width/2, cosine_means, width, yerr=cosine_sems, capsize=5, label="Cosine Error") # Use sems

#     plt.xlabel("Lambda Configuration (Design, Estimation)")
#     plt.ylabel("Error")
#     plt.xticks(x, labels, rotation=45, ha="right") # Rotate labels for better readability
#     plt.title("Error vs Lambda Configuration (Design & Estimation)")
#     plt.legend()
#     plt.tight_layout() # Adjust layout to prevent labels overlapping


def plot_feedback_results(feedback_results):
    # Create a grouped bar chart comparing both metrics for each algorithm.
    labels = list(feedback_results.keys())
    
    preference_means = []
    preference_sems = [] # Changed from stds to sems
    cosine_means = []
    cosine_sems = [] # Changed from stds to sems

    for alg in labels:
        pref_errors = feedback_results[alg]["preference_error"]
        cos_errors = feedback_results[alg]["cosine_error"]
        
        n_pref = len(pref_errors)
        n_cos = len(cos_errors)
        
        preference_means.append(np.mean(pref_errors) if n_pref > 0 else 0)
        preference_sems.append((np.std(pref_errors) / np.sqrt(n_pref)) if n_pref > 1 else 0) # Calculate SEM
        
        cosine_means.append(np.mean(cos_errors) if n_cos > 0 else 0)
        cosine_sems.append((np.std(cos_errors) / np.sqrt(n_cos)) if n_cos > 1 else 0) # Calculate SEM

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, preference_means, width, yerr=preference_sems, capsize=5, label="Preference Error") # Use sems
    plt.bar(x + width/2, cosine_means, width, yerr=cosine_sems, capsize=5, label="Cosine Error") # Use sems

    plt.xlabel("Algorithm")
    plt.ylabel("Error")
    plt.xticks(x, labels)
    plt.title("Algorithm Comparison on Error Metrics")
    plt.legend()
    plt.tight_layout()

def plot_feedback_over_episodes(data, episodes_x_axis, error_type_to_plot, model_name_filter=None, save_plots=False, output_dir="."): # Added save_plots, output_dir
    """
    Plots feedback experiment results for a specific error type (errors vs. number of episodes) as a line graph.

    Args:
        data (dict): Processed data in the format {alg_name: {error_type: {episode: [values]}}}.
        episodes_x_axis (list): Sorted list of unique episode values for the x-axis.
        error_type_to_plot (str): The specific error type to plot (e.g., "preference_error").
    """
    plt.figure(figsize=(12, 8)) # Increased height from 6 to 8
    algorithms = sorted(data.keys()) # e.g., ["Design", "Random"]
    
    # Define distinct colors, markers, and linestyles
    # Using more distinct styles for preference vs cosine when they were on the same plot.
    # Now, they are separate, so we can simplify if needed, but keeping distinctness for clarity.
    plot_styles = {
        "Design": {
            "preference_error": {"color": "darkblue", "marker": "o", "linestyle": "-", "linewidth": 2.5},
            "cosine_error": {"color": "darkgreen", "marker": "s", "linestyle": "-", "linewidth": 2.5}
        },
        "Random": {
            "preference_error": {"color": "crimson", "marker": "X", "linestyle": "--", "linewidth": 2.5},
            "cosine_error": {"color": "darkorange", "marker": "D", "linestyle": "--", "linewidth": 2.5}
        }
    }
    
    font_size_axis = 22  # Increased from 18
    font_size_title = 24 # Increased from 20
    font_size_ticks_legend = 20 # Increased from 16

    for alg in algorithms:
        means = []
        sems = []
        current_episodes_for_plot = []

        for ep in episodes_x_axis:
            values = data.get(alg, {}).get(error_type_to_plot, {}).get(ep, [])
            if values:
                means.append(np.mean(values))
                sems.append((np.std(values) / np.sqrt(len(values))) if len(values) > 1 else 0)
                current_episodes_for_plot.append(ep)
        
        if means: # Only plot if there's data for this combination
            style = plot_styles.get(alg, {}).get(error_type_to_plot, {})
            label = f"{alg}" # Simpler label as error type is in title
            
            plt.plot(current_episodes_for_plot, means, label=label, 
                     color=style.get("color"), 
                     linestyle=style.get("linestyle", "-"), 
                     marker=style.get("marker"),
                     linewidth=style.get("linewidth", 2)) # Apply linewidth
            plt.fill_between(current_episodes_for_plot, 
                             np.array(means) - np.array(sems), 
                             np.array(means) + np.array(sems), 
                             color=style.get("color"), alpha=0.2)

    plt.xlabel("Number of Episodes", fontsize=font_size_axis)
    # Specific Y-axis label
    y_label = f"{error_type_to_plot.replace('_', ' ').title()} Rate"
    plt.ylabel(y_label, fontsize=font_size_axis)
    
    # Set Y-axis limits based on error type
    if error_type_to_plot == "preference_error":
        plt.ylim(0, 0.5)
    elif error_type_to_plot == "cosine_error":
        plt.ylim(0, 1.0)
    
    # Specific title
    plot_title = f"{error_type_to_plot.replace('_', ' ').title()} vs. Number of Episodes"
    if model_name_filter:
        plot_title += f" (Model: {model_name_filter})"
    plt.title(plot_title, fontsize=font_size_title)
    
    if episodes_x_axis: # Avoid error if episodes_x_axis is empty
        plt.xticks(episodes_x_axis, fontsize=font_size_ticks_legend) 
    plt.yticks(fontsize=font_size_ticks_legend)
    plt.legend(loc='best', fontsize=font_size_ticks_legend)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    if save_plots:
        error_suffix = "pref" if "preference" in error_type_to_plot else "cosine"
        filename = os.path.join(output_dir, f"plot_feedback_episodes_{error_suffix}{'_' + model_name_filter if model_name_filter else ''}.png")
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close()

def plot_feedback_comparison_bar(data, episode_filter, title_suffix="", model_name_filter=None, save_plots=False, output_dir="."): # Added save_plots, output_dir
    """
    Plots feedback experiment results as a bar chart for a single episode configuration.

    Args:
        data (dict): Processed data in the format {alg_name: {error_type: {episode: [values]}}}.
        episode_filter (int): The specific episode value to plot (or -1 for legacy).
        title_suffix (str): Suffix to add to the plot title.
    """
    labels = sorted(data.keys()) # e.g., ["Design", "Random"]
    preference_means = []
    preference_sems = []
    cosine_means = []
    cosine_sems = []

    for alg in labels:
        pref_errors = data.get(alg, {}).get("preference_error", {}).get(episode_filter, [])
        cos_errors = data.get(alg, {}).get("cosine_error", {}).get(episode_filter, [])
        
        n_pref = len(pref_errors)
        n_cos = len(cos_errors)
        
        preference_means.append(np.mean(pref_errors) if n_pref > 0 else np.nan) # Use nan for missing data
        preference_sems.append((np.std(pref_errors) / np.sqrt(n_pref)) if n_pref > 1 else 0)
        
        cosine_means.append(np.mean(cos_errors) if n_cos > 0 else np.nan) # Use nan for missing data
        cosine_sems.append((np.std(cos_errors) / np.sqrt(n_cos)) if n_cos > 1 else 0)

    # Filter out algorithms for which no data was found for this episode_filter
    valid_indices = [i for i, (pm, cm) in enumerate(zip(preference_means, cosine_means)) if not (np.isnan(pm) and np.isnan(cm))]
    if not valid_indices:
        print(f"No data found for episode filter {episode_filter} in plot_feedback_comparison_bar.")
        return

    labels = [labels[i] for i in valid_indices]
    preference_means = [preference_means[i] for i in valid_indices]
    preference_sems = [preference_sems[i] for i in valid_indices]
    cosine_means = [cosine_means[i] for i in valid_indices]
    cosine_sems = [cosine_sems[i] for i in valid_indices]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, preference_means, width, yerr=preference_sems, capsize=5, label="Preference Error", color='tab:blue')
    rects2 = ax.bar(x + width/2, cosine_means, width, yerr=cosine_sems, capsize=5, label="Cosine Error", color='tab:orange')

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Error Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    title = f"Algorithm Comparison on Error Metrics{title_suffix}"
    if model_name_filter:
        title += f" (Model: {model_name_filter})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7, axis='y')
    fig.tight_layout()
    if save_plots:
        ep_suffix = f"_ep{episode_filter}" if episode_filter != -1 else "_legacy"
        filename = os.path.join(output_dir, f"plot_feedback_bar{ep_suffix}{'_' + model_name_filter if model_name_filter else ''}.png")
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.close(fig) # Close the specific figure

def plot_results(directory, cli_model_name=None, save_plots=False): # Added save_plots
    output_dir = directory # Save plots in the same directory as the data
    pattern = os.path.join(directory, "*.json")
    files = glob.glob(pattern)

    target_model_name = cli_model_name # Use CLI model name if provided

    if target_model_name is None:
        # Attempt to auto-detect single model from config files
        config_files = glob.glob(os.path.join(directory, "config_resolved-*.yaml"))
        detected_model_names = set()
        if config_files:
            # print(f"Found {len(config_files)} config files for auto-detection: {config_files}") # Removed this line
            for config_file_path in config_files:
                try:
                    with open(config_file_path, 'r') as f_cfg:
                        config_data = yaml.safe_load(f_cfg)
                    # Access experiment.scorer_model, which can be str or list
                    scorer_model_val = config_data.get('experiment', {}).get('scorer_model')
                    if isinstance(scorer_model_val, str):
                        detected_model_names.add(scorer_model_val)
                    elif isinstance(scorer_model_val, list) and len(scorer_model_val) == 1 and isinstance(scorer_model_val[0], str):
                        detected_model_names.add(scorer_model_val[0])
                    elif isinstance(scorer_model_val, list) and len(scorer_model_val) > 1:
                        # If multiple models are listed in a single config, this implies multi-model run, not a single target.
                        # Add a special marker or handle as "multiple defined"
                        detected_model_names.add("MULTIPLE_MODELS_IN_CONFIG") 
                        break # Stop if a config explicitly lists multiple models
                except Exception as e:
                    print(f"Warning: Could not load or parse config {config_file_path}: {e}")
            
            if len(detected_model_names) == 1 and "MULTIPLE_MODELS_IN_CONFIG" not in detected_model_names:
                target_model_name = list(detected_model_names)[0]
                print(f"Auto-detected single model name: {target_model_name}")
            elif len(detected_model_names) > 1 or "MULTIPLE_MODELS_IN_CONFIG" in detected_model_names:
                print(f"Multiple or ambiguous models detected in configs: {detected_model_names}. Will plot aggregated/general results if available.")
                target_model_name = None # Ensure it's None if ambiguous
            else:
                print("No single model name auto-detected from configs. Will plot aggregated/general results if available.")
        else:
            print("No config files found for auto-detection. Will plot aggregated/general results if available.")
    else:
        print(f"Using model name from CLI: {target_model_name}")

    # Dictionaries for different experiment types based on filename parsing
    results_by_type = {
        "lambda": {},
        "v_comparison": {},
        "frequency": {}, # Old frequency key, might be unused now
        "rounds": {},
        "design_frequency": {}, # New key for design frequency results
        "lambda_sparsity": {},  # Added key for lambda_sparsity experiments
        # "lambda_dsn_est": {}, # Removed as this experiment type is no longer used
        "feedback": {}          # For feedback comparison experiments (dsn-mult vs rand-mult etc.)
    }

    for f in files:
        try:
            exp_type, key_info, model_name_from_file = parse_filename(f)
        except Exception as e:
            print(f"Warning: Could not parse filename {os.path.basename(f)}: {e}")
            continue

        if exp_type == "unknown":
            continue
            
        # Filter files based on target_model_name
        if target_model_name is not None: # Plotting for a specific model (e.g., "sunny")
            # If the file has a model name, it must match the target_model_name.
            # If the file does NOT have a model name (model_name_from_file is None),
            # we include it because we have a specific target_model_name,
            # implying these "general" files belong to this target model.
            if model_name_from_file is not None and model_name_from_file != target_model_name:
                continue # Skip if file has a model name, but it's not the one we're looking for.
            # Otherwise, (model_name_from_file is None OR model_name_from_file == target_model_name), process the file.
        else: # Plotting aggregated/general results (target_model_name is None)
            # Only include files that also have no model name (old general/aggregated files).
            if model_name_from_file is not None:
                continue # Skip model-specific files

        val = safe_load_data(f)
        if val is None:
            # print(f"Warning: Could not load data from {f}. Skipping.") # Removed this line
            continue

        if exp_type == "feedback":
            # key_info is (alg_code, feed_code, episodes)
            # We will store it directly as the key in results_by_type["feedback"]
            # The processing into alg_name, error_type, episode will happen in plot_feedback_results
            if key_info not in results_by_type["feedback"]:
                results_by_type["feedback"][key_info] = {"preference_error": [], "cosine_error": []}
            
            if isinstance(val, dict):
                if "preference_error" in val:
                    results_by_type["feedback"][key_info]["preference_error"].append(val["preference_error"])
                if "cosine_error" in val:
                    results_by_type["feedback"][key_info]["cosine_error"].append(val["cosine_error"])

        elif exp_type in results_by_type: # Handle all other types
            if key_info not in results_by_type[exp_type]:
                results_by_type[exp_type][key_info] = []
            results_by_type[exp_type][key_info].append(val)
        else:
            # This case should ideally not be reached if parse_filename is comprehensive
            # or returns "unknown" for unhandled patterns.
            print(f"Warning: Unrecognized experiment type '{exp_type}' for file {os.path.basename(f)} with key {key_info}")

    # Plot non-feedback experiments (excluding design_frequency and feedback)
    for exp_type in results_by_type:
        if exp_type not in ["design_frequency", "feedback"] and results_by_type[exp_type]:
            plot_results_with_type(results_by_type[exp_type], exp_type, model_name_filter=target_model_name, save_plots=save_plots, output_dir=output_dir)
 
    # Plot design frequency results separately
    if results_by_type["design_frequency"]:
        plot_design_frequency_results(results_by_type["design_frequency"], model_name_filter=target_model_name, save_plots=save_plots, output_dir=output_dir)

    # Plot lambda_dsn_est results separately - This section is removed as the function and type are removed.
    # if results_by_type["lambda_dsn_est"]:
    #     plot_lambda_dsn_est_results(results_by_type["lambda_dsn_est"])
 
    # Plot feedback results using the new dispatcher
    if results_by_type["feedback"]:
        # New dispatcher function for feedback plots
        # This function will decide whether to call plot_feedback_over_episodes or plot_feedback_comparison_bar
        
        # Pre-process data for the feedback plotting functions
        processed_feedback_data = {} # To store {alg_name: {error_type: {episode: [values]}}}
        all_episode_values = set()
        alg_map = {"dsn": "Design", "rand": "Random"}
        # feed_map = {"mult": "Multinomial", "num": "Numerical"} # Not currently used for plot differentiation

        for key_tuple, error_data_dict in results_by_type["feedback"].items():
            alg_code, _, episodes = key_tuple # feed_code is part of key_tuple but not used for grouping lines/bars here
            alg_name = alg_map.get(alg_code, alg_code)
            
            all_episode_values.add(episodes)

            if alg_name not in processed_feedback_data:
                processed_feedback_data[alg_name] = {"preference_error": {}, "cosine_error": {}}

            # error_data_dict is like {"preference_error": [0.1, 0.2], "cosine_error": [0.3, 0.4]}
            for error_type, values_list in error_data_dict.items():
                if error_type in processed_feedback_data[alg_name]: # Should be "preference_error" or "cosine_error"
                    if episodes not in processed_feedback_data[alg_name][error_type]:
                        processed_feedback_data[alg_name][error_type][episodes] = []
                    processed_feedback_data[alg_name][error_type][episodes].extend(values_list)
        
        unique_episodes_for_plot = sorted([ep for ep in all_episode_values if ep != -1]) # Actual episode values for line plot x-axis
        has_legacy_data = -1 in all_episode_values

        if len(unique_episodes_for_plot) > 1:
            # Plot Preference Error
            plot_feedback_over_episodes(processed_feedback_data, unique_episodes_for_plot, "preference_error", model_name_filter=target_model_name, save_plots=save_plots, output_dir=output_dir)
            if not save_plots:
                plt.show(block=False) # Show first plot, allow script to continue for the next one
            
            # Plot Cosine Error
            plot_feedback_over_episodes(processed_feedback_data, unique_episodes_for_plot, "cosine_error", model_name_filter=target_model_name, save_plots=save_plots, output_dir=output_dir)
        elif len(unique_episodes_for_plot) == 1:
            single_episode_val = unique_episodes_for_plot[0]
            plot_feedback_comparison_bar(processed_feedback_data, episode_filter=single_episode_val, title_suffix=f" (Episodes: {single_episode_val})", model_name_filter=target_model_name, save_plots=save_plots, output_dir=output_dir)
        elif has_legacy_data and not unique_episodes_for_plot: # Only legacy data
            plot_feedback_comparison_bar(processed_feedback_data, episode_filter=-1, title_suffix=" (Legacy Format)", model_name_filter=target_model_name, save_plots=save_plots, output_dir=output_dir)
        else:
            if results_by_type["feedback"]: # Check if there was any feedback data at all
                 print("No feedback data suitable for plotting (e.g., only one legacy data point per alg/error type).")

    if not save_plots:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument('directory', help='Directory containing experiment result .json files.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Specific model name to filter results for. If not provided, attempts to auto-detect or plots aggregated results.')
    parser.add_argument('--save', action='store_true', help='Save plots to files instead of displaying them.')
    args = parser.parse_args()
    plot_results(args.directory, args.model_name, args.save)
