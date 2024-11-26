import numpy as np
import matplotlib.pyplot as plt

R = 5
ALGS = ["alg","rand","greedy"]
prefix = "../results/"
suffix = ".txt"
d = {}
for alg in ALGS:
    d[alg] = (0,0)

for alg in ALGS:
    vals = []
    for i in range(R):
        path = prefix + alg +"-"+str(i+1)+ suffix
        val = np.loadtxt(path)
        vals.append(val)
    d[alg] = (np.mean(vals),np.std(vals))

means = [d[alg][0] for alg in ALGS]
std_devs = [d[alg][1] for alg in ALGS]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(ALGS, means, yerr=std_devs, capsize=5, color='skyblue', edgecolor='black')
plt.xlabel("Algorithms")
plt.ylabel("MSE of Aesthetics model")
plt.title("Algorithms and their MSE with Error Bars")
plt.show()
