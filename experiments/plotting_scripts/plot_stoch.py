import numpy as np
import matplotlib.pyplot as plt

color = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:brown", "tab:purple"]

for j, p in enumerate([0.1, 0.3, 0.5]):

    methods = ["adaptive-1-prob-"+str(p),"adaptive-prob-"+str(p),"tracking-prob-"+str(p), "density-prob-"+str(p), "mixture-prob-"+str(p), "random-prob-"+str(p)]

    name = "stochastic_grid/results/opt+"+str(np.round(float(p),2))+".txt"
    opt = np.loadtxt(name)

    for index, method in enumerate(methods):

        vals = []
        for i in range(10):
            name = "stochastic_grid/results/"+ method + "-" + str(i+1)  + ".txt"
            val = np.array(np.loadtxt(name))
            vals.append(val)
        print (np.array(vals).shape)
        median = np.median(opt-np.array(vals),axis = 0)
        q10 = np.quantile(opt - np.array(vals), q=0.1, axis=0)
        q90 = np.quantile(opt - np.array(vals), q=0.9, axis=0)
        xaxis = np.arange(1, median.shape[0] + 1, 1)
        plt.plot(xaxis, median, color=color[index], label="-".join(method.split("-")[:-2]))
        plt.fill_between(xaxis, q10, q90, alpha=0.3, color=color[index])

    plt.plot(xaxis, 2 ** 4 / np.sqrt(xaxis), 'k--', label='1/\u221Ax')
    plt.plot(xaxis, 2 ** 4 / xaxis, 'k:', label='1/x')
    plt.xlabel("Episodes [t]", fontsize="xx-large")
    plt.ylabel("$F(p_t) - F(p^*)$", fontsize="xx-large")
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.grid("--", color = 'gray', alpha=0.5)
    plt.yticks(fontsize="x-large")
    plt.xticks(fontsize="x-large")
    plt.legend(fontsize="x-large", borderpad=0.1, labelspacing=0.1)
    plt.savefig("figs/stoch_grids-known"+str(j)+".png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()