import numpy as np
import matplotlib.pyplot as plt

methods = ["adaptive-1","adaptive","tracking", "density", "random"]
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]

labels = ["adaptive-1","adaptive","tracking", "non-adaptive", "random"]
NAMES = ["LINEAR","EXACT","TRACKING","RESAMPLING","RANDOM"]

name = "beilschmiedia/results/opt.txt"
opt = np.loadtxt(name)
y_lim = [1,2**9]
ax = plt.axes()
for index, method in enumerate(methods):
    vals = []
    for i in range(10):
        name = "beilschmiedia/results/"+ method + "-" + str(i+1)  + ".txt"
        val = np.array(np.loadtxt(name))
        vals.append(val)
    median = np.median(opt-np.array(vals),axis = 0)
    q10 = np.quantile(opt-np.array(vals),q=0.1, axis = 0)
    q90 = np.quantile(opt-np.array(vals),q=0.9, axis =0)
    xaxis = np.arange(1,median.shape[0]+1,1)
    plt.plot(xaxis, median, color = colors[index], label = NAMES[index])
    plt.fill_between(xaxis, q10, q90, alpha = 0.3, color = colors[index])

plt.xlim([2**0, 2**7])
plt.plot(xaxis, 2 ** 9 / np.sqrt(xaxis), 'k', label='1/\u221At')
plt.plot(xaxis, 2 ** 9 / xaxis, 'k:', label='1/t')
plt.plot(xaxis, 2 ** 9  / xaxis**2, 'k-.', label='$1/t^2$', linewidth = 2)

plt.grid("--", color = 'gray', alpha=0.5)
plt.xlabel("Episodes [t]", fontsize="xx-large")
plt.ylabel("$F(p_t) - F(p^*)$", fontsize="xx-large")
plt.xscale('log', base =2 )
plt.yscale('log', base =2 )
plt.yticks(fontsize="x-large")
plt.xticks(fontsize="x-large")
plt.legend(fontsize="x-large", borderpad=0.1, labelspacing=0.1)
ax.patch.set_facecolor('grey')
ax.patch.set_alpha(0.2)
plt.ylim(y_lim)
plt.savefig("figs/bels-known.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)
plt.show()

methods = ["adaptive-un-1","adaptive-un","tracking-un", "density-un", "random-un"]

name = "beilschmiedia/results/un-opt.txt"
opt = np.loadtxt(name)

ax = plt.axes()
for index, method in enumerate(methods):
    vals = []
    for i in range(10):
        name = "beilschmiedia/results/"+ method + "-" + str(i+1)  + ".txt"
        val = np.loadtxt(name)
        vals.append(val)
    median = np.median(opt-np.array(vals),axis = 0)
    q10 = np.quantile(opt-np.array(vals),q=0.1, axis = 0)
    q90 = np.quantile(opt-np.array(vals),q=0.9, axis =0)
    xaxis = np.arange(1,median.shape[0]+1,1)
    plt.plot(xaxis, median, color = colors[index], label = NAMES[index])
    plt.fill_between(xaxis, q10, q90, alpha = 0.3, color = colors[index])

plt.xlim([2**0, 2**7])
plt.plot(xaxis, 2 ** 9 / np.sqrt(xaxis), 'k', label='1/\u221At')
plt.plot(xaxis, 2 ** 9 / xaxis, 'k:', label='1/t')
plt.plot(xaxis, 2 ** 9  / xaxis**2, 'k-.', label='$1/t^2$', linewidth = 2)

plt.grid("--", color = 'gray', alpha=0.5)
plt.xlabel("Episodes [t]", fontsize="xx-large")
plt.ylabel("$F(p_t) - F(p^*)$", fontsize="xx-large")
plt.xscale('log', base =2 )
plt.yscale('log', base =2 )
plt.yticks(fontsize="x-large")
plt.xticks(fontsize="x-large")
plt.legend(fontsize="x-large", borderpad=0.1, labelspacing=0.1)
ax.patch.set_facecolor('grey')
ax.patch.set_alpha(0.2)
plt.savefig("figs/bels-unknown.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)
plt.show()
