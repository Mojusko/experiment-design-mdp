import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import json

def parse_filename(filename):
    base = os.path.basename(filename)
    # Check for other experiment types first.
    if "withV" in base or "noV" in base:
        v_type = "With V" if "withV" in base else "No V"
        return ("v_comparison", v_type)
    elif "lambda" in base:
        parts = base.split('-')
        lambda_val = float(parts[-2])
        return ("lambda", lambda_val)
    elif "freq" in base:
        parts = base.split('-')
        freq_val = int(parts[-2])
        return ("frequency", freq_val)
    elif "rounds" in base:
        parts = base.split('-')
        rounds_val = int(parts[-2])
        return ("rounds", rounds_val)
    else:
        # For feedback experiments, assume file names like "grd-multinomial-1.txt" or "rnd-sample-1.txt"
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
        means = [np.mean(valid_data[k]) for k in valid_keys]
        stds = [np.std(valid_data[k]) if len(valid_data[k]) > 1 else 0 for k in valid_keys]

        plt.figure(figsize=(10, 6))
        plt.bar([str(x) for x in valid_keys], means, yerr=stds, capsize=5)

        if plot_type == "lambda":
            plt.xlabel("Lambda Value")
        elif plot_type == "frequency":
            plt.xlabel("Estimation Frequency")
        elif plot_type == "rounds":
            plt.xlabel("Number of Rounds")
        elif plot_type == "v_comparison":
            plt.xlabel("Design Matrix Type")
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
    preference_means = [np.mean([d["preference_error"] for d in results[alg] if isinstance(d, dict) and "preference_error" in d]) 
                        if any(isinstance(d, dict) and "preference_error" in d for d in results[alg]) else 0 
                        for alg in labels]
    preference_stds = [np.std([d["preference_error"] for d in results[alg] if isinstance(d, dict) and "preference_error" in d]) 
                       if any(isinstance(d, dict) and "preference_error" in d for d in results[alg]) else 0 
                       for alg in labels]
    cosine_means = [np.mean([d["cosine_error"] for d in results[alg] if isinstance(d, dict) and "cosine_error" in d]) 
                    if any(isinstance(d, dict) and "cosine_error" in d for d in results[alg]) else 0 
                    for alg in labels]
    cosine_stds = [np.std([d["cosine_error"] for d in results[alg] if isinstance(d, dict) and "cosine_error" in d]) 
                   if any(isinstance(d, dict) and "cosine_error" in d for d in results[alg]) else 0 
                   for alg in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, preference_means, width, yerr=preference_stds, capsize=5, label="Preference Error")
    plt.bar(x + width/2, cosine_means, width, yerr=cosine_stds, capsize=5, label="Cosine Error")

    plt.xlabel("Design Matrix Type" if plot_type == "v_comparison" else plot_type.capitalize())
    plt.ylabel("Error")
    plt.xticks(x, labels)
    plt.title(f"{plot_type.capitalize()} Comparison on Error Metrics")
    plt.legend()
    plt.tight_layout()

def plot_feedback_results(feedback_results):
    # Create a grouped bar chart comparing both metrics for each algorithm.
    labels = list(feedback_results.keys())
    preference_means = [np.mean(feedback_results[alg]["preference_error"]) for alg in labels]
    preference_stds = [np.std(feedback_results[alg]["preference_error"]) for alg in labels]
    cosine_means = [np.mean(feedback_results[alg]["cosine_error"]) for alg in labels]
    cosine_stds = [np.std(feedback_results[alg]["cosine_error"]) for alg in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, preference_means, width, yerr=preference_stds, capsize=5, label="Preference Error")
    plt.bar(x + width/2, cosine_means, width, yerr=cosine_stds, capsize=5, label="Cosine Error")

    plt.xlabel("Algorithm")
    plt.ylabel("Error")
    plt.xticks(x, labels)
    plt.title("Algorithm Comparison on Error Metrics")
    plt.legend()
    plt.tight_layout()

def plot_results(directory):
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)

    # Dictionaries for non-feedback experiments.
    results_by_type = {
        "lambda": {},
        "v_comparison": {},
        "frequency": {},
        "rounds": {}
    }
    # For feedback experiments (algorithm comparison), group by algorithm.
    feedback_results = {}

    for f in files:
        exp_type, key = parse_filename(f)
        val = safe_load_data(f)

        if exp_type == "feedback":
            # key is a tuple: (alg_type, feedback_type). We group by algorithm only.
            alg = key[0]
            # Map shorthand names to full algorithm names.
            alg_map = {"grd": "Greedy", "rnd": "Random"}
            alg_name = alg_map.get(alg, alg)
            if alg_name not in feedback_results:
                feedback_results[alg_name] = {"preference_error": [], "cosine_error": []}
            if val is not None and isinstance(val, dict):
                if "preference_error" in val:
                    feedback_results[alg_name]["preference_error"].append(val["preference_error"])
                if "cosine_error" in val:
                    feedback_results[alg_name]["cosine_error"].append(val["cosine_error"])
        else:
            # For other experiments, use the existing grouping.
            if key not in results_by_type[exp_type]:
                results_by_type[exp_type][key] = []
            if val is not None:
                results_by_type[exp_type][key].append(val)

    # Plot non-feedback experiments.
    for exp_type in results_by_type:
        if results_by_type[exp_type]:
            plot_results_with_type(results_by_type[exp_type], exp_type)

    # Plot the feedback (algorithm comparison) results.
    if feedback_results:
        plot_feedback_results(feedback_results)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experiment results.")
    parser.add_argument('directory', help='Directory containing experiment result .txt files.')
    args = parser.parse_args()
    plot_results(args.directory)
