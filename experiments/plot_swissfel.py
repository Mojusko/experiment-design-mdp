import numpy as np
import matplotlib.pyplot as plt

repeats = 10
color = ["tab:blue", "tab:orange"]
methods = ["con","worst"]
names = ['MDP-BO','state-less BO']
for index, method in enumerate(methods):

    vals = []
    for i in range(repeats):
        name = "swissfel/results/"+ method + "-" + str(i+1)  + ".txt"
        val = np.array(np.loadtxt(name))
        vals.append(val)
    print (np.array(vals).shape)
    median = np.median(np.array(vals),axis = 0)
    q10 = np.quantile(np.array(vals), q=0.1, axis=0)
    q90 = np.quantile(np.array(vals), q=0.9, axis=0)
    xaxis = np.arange(1, median.shape[0] + 1, 1)
    plt.plot(xaxis, median, color=color[index], label=names[index])
    plt.fill_between(xaxis, q10, q90, alpha=0.3, color=color[index])

#plt.plot(xaxis, 2 ** 4 / np.sqrt(xaxis), 'k--', label='1/\u221Ax')
#plt.plot(xaxis, 2 ** 4 / xaxis, 'k:', label='1/x')
plt.xlabel("Horizon [h]", fontsize="xx-large")
plt.ylabel("Inference Regret", fontsize="xx-large")
#plt.xscale('log', base=2)
#plt.yscale('log', base=2)
plt.grid("--", color = 'gray', alpha=0.5)
plt.yticks(fontsize="x-large")
plt.xticks(fontsize="x-large")
plt.legend(fontsize="x-large", borderpad=0.1, labelspacing=0.1)
#plt.legend(loc = 'lower left',fontsize="x-large")
plt.savefig("figs/swissfel.png", dpi=100, bbox_inches='tight', pad_inches=0)
plt.show()