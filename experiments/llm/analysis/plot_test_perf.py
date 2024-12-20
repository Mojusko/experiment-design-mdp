import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def parse_filename(filename):
    # Extract alg type and feedback type from filename like "alg-numerical-1.txt"
    base = os.path.basename(filename)
    alg_type, feedback_type, _ = base.rsplit('-', 2)
    return alg_type, feedback_type

def plot_results(directory):
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)
    
    # Group results by algorithm and feedback type
    results = {}
    for f in files:
        alg_type, feedback_type = parse_filename(f)
        key = f"{alg_type}-{feedback_type}"
        if key not in results:
            results[key] = []
        val = np.loadtxt(f)
        results[key].append(val)
    
    # Calculate means and stds
    alg_keys = sorted(results.keys())
    means = [np.mean(results[k]) for k in alg_keys]
    stds = [np.std(results[k]) for k in alg_keys]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(alg_keys, means, yerr=stds, capsize=5)
    plt.xlabel("Algorithm-Feedback Type")
    plt.ylabel("Preference Misalignment Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing results with different algs / feedback models.')
    args = parser.parse_args()
    plot_results(args.directory)
