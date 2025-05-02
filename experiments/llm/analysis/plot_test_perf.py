import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import json
import re # Import re for pattern matching

def parse_filename(filename):
    base = os.path.basename(filename)
    # Check for other experiment types first.
    if "withV" in base or "noV" in base:
        v_type = "With V" if "withV" in base else "No V"
        return ("v_comparison", v_type)
    # Match new lambda format with dsn and est: metrics-lambda-dsn-mult-mul-dsn0.1-est1000-6.json
    elif match := re.search(r"dsn([\d.]+)-est([\d.]+)-(\d+)\.json$", base):
        dsn_val = float(match.group(1))
        est_val = float(match.group(2))
        # seed = int(match.group(3)) # Seed not used for grouping key
        return ("lambda_dsn_est", (dsn_val, est_val))
    # Match old single lambda format: metrics-mul-lambda-0.1-1.json or metrics-mul-0.1-1.json
    elif "lambda" in base or re.search(r"mul-([\d.]+)-(\d+)\.json$", base) or re.search(r"num-([\d.]+)-(\d+)\.json$", base):
        # Extract lambda value robustly
        match_lambda = re.search(r"lambda-([\d.]+)", base)
        match_mul = re.search(r"mul-([\d.]+)-(\d+)\.json$", base)
        match_num = re.search(r"num-([\d.]+)-(\d+)\.json$", base)
        if match_lambda:
            lambda_val = float(match_lambda.group(1))
        elif match_mul:
            lambda_val = float(match_mul.group(1))
        elif match_num:
            lambda_val = float(match_num.group(1))
        else:
             # Fallback if pattern is unexpected, try finding the last number before the seed
             parts = base.split('-')
             try:
                 # Assume format like *-<lambda>-<seed>.json
                 lambda_val = float(parts[-2])
             except (ValueError, IndexError):
                 print(f"Warning: Could not extract lambda from fallback pattern in {base}. Skipping.")
                 return ("unknown", base) # Return an identifiable unknown type
        return ("lambda", lambda_val)
    # Match design frequency filenames like metrics-design-freq-dsn-mult-ep25-df10-1.json
    elif match := re.search(r"ep(\d+)-df(\d+)-(\d+)\.json$", base):
        episodes = int(match.group(1))
        frequency = int(match.group(2))
        # seed = int(match.group(3)) # Seed not used for grouping key
        return ("design_frequency", (episodes, frequency))
    elif "rounds" in base:
        parts = base.split('-')
        rounds_val = int(parts[-2])
        return ("rounds", rounds_val)
    else:
        # For feedback experiments, assume file names like "dsn-multinomial-1.txt" or "rnd-sample-1.txt"
        alg_type, feedback_type, _ = base.rsplit('-', 2)
        return ("feedback", (alg_type, feedback_type))

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
            data_dict = json.load(f)
        # If both keys are present, return the entire dictionary.
        if "preference_error" in data_dict and "cosine_error" in data_dict:
            return data_dict
        elif "cosine_error" in data_dict:
            return {"cosine_error": data_dict["cosine_error"]}
        else:
            # Return the first numeric value encountered.
            for value in data_dict.values():
                if isinstance(value, (int, float)):
                    return value
    except Exception:
        pass

    return None

def plot_results_with_type(results, plot_type):
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
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        # Handle dictionary data (like v_comparison with multiple metrics)
        plot_comparison_results(valid_data, plot_type)

def plot_comparison_results(results, plot_type):
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
    plt.xticks(x, labels)
    plt.title(f"{plot_type.capitalize()} Comparison on Error Metrics")
    plt.legend()
    plt.tight_layout()

def plot_design_frequency_results(design_freq_results):
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
    plt.xticks(x, x_labels, rotation=45, ha="right") # Rotate labels for better readability
    plt.title("Design Frequency Experiment Results")
    plt.legend()
    plt.tight_layout()


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

def plot_results(directory):
    pattern = os.path.join(directory, "*.json")
    files = glob.glob(pattern)

    # Dictionaries for different experiment types based on filename parsing
    results_by_type = {
        "lambda": {},
        "v_comparison": {},
        "frequency": {}, # Old frequency key, might be unused now
        "rounds": {},
        "design_frequency": {}, # New key for design frequency results
        "lambda_dsn_est": {},   # New key for dsn/est lambda experiments
        "feedback": {}          # For feedback comparison experiments (dsn-mult vs rand-mult etc.)
    }

    for f in files:
        try:
            exp_type, key = parse_filename(f)
        except Exception as e:
            print(f"Warning: Could not parse filename {os.path.basename(f)}: {e}")
            continue # Skip this file

        val = safe_load_data(f)

        if exp_type == "feedback":
            alg_key = key[0] # alg_type from tuple (e.g., 'dsn', 'rand')
            alg_map = {"dsn": "Design", "rand": "Random"}
            alg_name = alg_map.get(alg_key, alg_key) # Use mapped name or original key

            # Initialize if first time seeing this algorithm
            if alg_name not in results_by_type["feedback"]:
                results_by_type["feedback"][alg_name] = {"preference_error": [], "cosine_error": []}

            # Append data if valid
            if val is not None and isinstance(val, dict):
                if "preference_error" in val:
                    results_by_type["feedback"][alg_name]["preference_error"].append(val["preference_error"])
                if "cosine_error" in val:
                    results_by_type["feedback"][alg_name]["cosine_error"].append(val["cosine_error"])
        elif exp_type in results_by_type: # Handle all other types
            if key not in results_by_type[exp_type]:
                results_by_type[exp_type][key] = []
            if val is not None:
                results_by_type[exp_type][key].append(val)
        else:
            print(f"Warning: Unrecognized experiment type '{exp_type}' for file {os.path.basename(f)}")

 
    # Plot non-feedback experiments (excluding design_frequency, feedback, and lambda_dsn_est)
    for exp_type in results_by_type:
        if exp_type not in ["design_frequency", "feedback", "lambda_dsn_est"] and results_by_type[exp_type]:
            plot_results_with_type(results_by_type[exp_type], exp_type)
 
    # Plot design frequency results separately
    if results_by_type["design_frequency"]:
        plot_design_frequency_results(results_by_type["design_frequency"])

    # Plot lambda_dsn_est results separately
    if results_by_type["lambda_dsn_est"]:
        plot_lambda_dsn_est_results(results_by_type["lambda_dsn_est"])
 
    # Plot feedback results
    if results_by_type["feedback"]:
        plot_feedback_results(results_by_type["feedback"])

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument('directory', help='Directory containing experiment result .json files.')
    args = parser.parse_args()
    plot_results(args.directory)
