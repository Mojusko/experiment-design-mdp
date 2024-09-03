import numpy as np
import matplotlib.pyplot as plt

methods = ["adaptive-1","adaptive", "tracking", "density", "mixture", "random"]
color = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:brown", "tab:purple"]

name = "classical_grid/results/opt.txt"
opt = np.loadtxt(name)

for index, method in enumerate(methods):
    vals = []
    for i in range(10):
        name = "classical_grid/results/"+ method + "-" + str(i+1)  + ".txt"
        val = np.array(np.loadtxt(name))
        vals.append(val)
    vals = np.array(vals)
    gap = opt - vals
    median = np.median(gap,axis = 0)
    q10 = np.quantile(gap, q=0.1, axis=0)
    q90 = np.quantile(gap, q=0.9, axis=0)
    xaxis = np.arange(1, median.shape[0] + 1, 1)
    plt.plot(xaxis, median, color=color[index], label=methods[index])
    plt.fill_between(xaxis, q10, q90, alpha=0.3, color=color[index])

plt.plot(xaxis, 2 ** 5 / np.sqrt(xaxis), 'k--', label='1/\u221Ax')
plt.plot(xaxis, 2 ** 5 / xaxis, 'k:', label='1/x')
plt.xlabel("Episodes [t]", fontsize="xx-large")
plt.ylabel("$F(p_t) - F(p^*)$", fontsize="xx-large")
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.grid("--", color = 'gray', alpha=0.5)
plt.yticks(fontsize="x-large")
plt.xticks(fontsize="x-large")
plt.legend(fontsize="x-large", borderpad=0.1, labelspacing=0.1)
#plt.gca().set_ylim([2**-8,2**6])
plt.savefig("figs/grids-known.png", dpi=100, bbox_inches='tight', pad_inches=0)
plt.show()
