
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import json  # Import JSON for parsing JSON dumps

def parse_filename(filename):
    base = os.path.basename(filename)
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

    # If plain text fails, try to load as a JSON dump.
    try:
        with open(filename, 'r') as f:
            data_dict = json.load(f)
        # Look for the "cosine_error" key as in the provided JSON example.
        if "cosine_error" in data_dict:
            return data_dict["cosine_error"]
        else:
            # If "cosine_error" is not present, return the first numeric value encountered.
            for value in data_dict.values():
                if isinstance(value, (int, float)):
                    return value
    except Exception:
        pass

    return None

def plot_results_with_type(results, plot_type):
    if not results:
        return

    keys = sorted(results.keys())
    valid_data = {k: [v for v in results[k] if v is not None] for k in keys}
    valid_keys = [k for k in keys if valid_data[k]]

    if not valid_keys:
        return

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
    elif plot_type == "feedback":
        plt.xlabel("Algorithm-Feedback Type")
    elif plot_type == "v_comparison":
        plt.xlabel("Design Matrix Type")

    plt.ylabel("Preference Misalignment Error")
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_results(directory):
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)

    results_by_type = {
        "lambda": {},
        "feedback": {},
        "v_comparison": {},
        "frequency": {},
        "rounds": {}
    }

    for f in files:
        exp_type, key = parse_filename(f)
        val = safe_load_data(f)

        if exp_type == "feedback":
            combined_key = f"{key[0]}-{key[1]}"
            if combined_key not in results_by_type[exp_type]:
                results_by_type[exp_type][combined_key] = []
            if val is not None:
                results_by_type[exp_type][combined_key].append(val)
        else:
            if key not in results_by_type[exp_type]:
                results_by_type[exp_type][key] = []
            if val is not None:
                results_by_type[exp_type][key].append(val)

    for exp_type in results_by_type:
        if results_by_type[exp_type]:
            plot_results_with_type(results_by_type[exp_type], exp_type)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing experiment results.')
    args = parser.parse_args()
    plot_results(args.directory)
