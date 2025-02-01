import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

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
    else:
        alg_type, feedback_type, _ = base.rsplit('-', 2)
        return ("feedback", (alg_type, feedback_type))

def plot_frequency_results(results):
    freq_vals = sorted(results.keys())
    means = [np.mean(results[k]) for k in freq_vals]
    stds = [np.std(results[k]) for k in freq_vals]

    plt.figure(figsize=(10, 6))
    plt.bar([str(x) for x in freq_vals], means, yerr=stds, capsize=5)
    plt.xlabel("Estimation Frequency")
    plt.ylabel("Preference Misalignment Error")
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_lambda_results(results):
    lambda_vals = sorted(results.keys())
    means = [np.mean(results[k]) for k in lambda_vals]
    stds = [np.std(results[k]) for k in lambda_vals]

    plt.figure(figsize=(10, 6))
    plt.bar([str(x) for x in lambda_vals], means, yerr=stds, capsize=5)
    plt.xlabel("Lambda Value")
    plt.ylabel("Preference Misalignment Error")
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_feedback_results(results):
    alg_keys = sorted(results.keys())
    means = [np.mean(results[k]) for k in alg_keys]
    stds = [np.std(results[k]) for k in alg_keys]

    plt.figure(figsize=(10, 6))
    plt.bar(alg_keys, means, yerr=stds, capsize=5)
    plt.xlabel("Algorithm-Feedback Type")
    plt.ylabel("Preference Misalignment Error")
    plt.xticks(rotation=45)
    plt.tight_layout()

def plot_v_results(results):
    keys = sorted(results.keys())
    means = [np.mean(results[k]) for k in keys]
    stds = [np.std(results[k]) for k in keys]

    plt.figure(figsize=(10, 6))
    plt.bar(keys, means, yerr=stds, capsize=5)
    plt.xlabel("Design Matrix Type")
    plt.ylabel("Preference Misalignment Error")
    plt.tight_layout()

def plot_results(directory):
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)
    
    lambda_results = {}
    feedback_results = {}
    v_results = {}
    frequency_results = {}
    
    for f in files:
        exp_type, key = parse_filename(f)
        val = np.loadtxt(f)
        
        if exp_type == "lambda":
            if key not in lambda_results:
                lambda_results[key] = []
            lambda_results[key].append(val)
        elif exp_type == "v_comparison":
            if key not in v_results:
                v_results[key] = []
            v_results[key].append(val)
        elif exp_type == "frequency":
            if key not in frequency_results:
                frequency_results[key] = []
            frequency_results[key].append(val)
        else:
            combined_key = f"{key[0]}-{key[1]}"
            if combined_key not in feedback_results:
                feedback_results[combined_key] = []
            feedback_results[combined_key].append(val)
    
    if lambda_results:
        plot_lambda_results(lambda_results)
    if feedback_results:
        plot_feedback_results(feedback_results)
    if v_results:
        plot_v_results(v_results)
    if frequency_results:
        plot_frequency_results(frequency_results)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing experiment results.')
    args = parser.parse_args()
    plot_results(args.directory)
