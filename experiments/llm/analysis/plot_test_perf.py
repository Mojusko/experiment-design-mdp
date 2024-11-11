import numpy as np
import matplotlib.pyplot as plt

R = 5
ALGS = ["alg","ran"]
prefix = "../results/"
suffix = ".txt"
d = {}
for alg in ALGS:
    d[alg] = (0,0)
for alg in ALGS:
    vals = []
    for i in range(R):
        path = prefix + alg +"-"+str(i)+"-" + suffix
        val = np.loadtxt(path)
    vals.append(val)
    d[alg] = (np.mean(vals),np.std(vals))

# plot histogram from the dictionary
fig, ax = plt.subplots()
ax.bar(d.keys(),d.values())
plt.show()

