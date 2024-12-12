import numpy as np
import torch
import matplotlib.pyplot as plt

methods = ["adaptive-un-1","adaptive-un", 'tracking-un', "density-un", "random-un"]
color = ["tab:blue", "tab:orange","tab:red", "tab:green", "tab:purple"]
labels = ["adaptive-1","adaptive","tracking", "non-adaptive", "random"]
#NAMES = ["ONE-STEP","EXACT","TRACKING","NON-ADAPTIVE","RANDOM"]
NAMES = ["LINEAR","EXACT","TRACKING","RESAMPLING","RANDOM"]


methods = ["adaptive-un",  "density-un", "random-un"]
color = ["tab:blue", "tab:green", "tab:purple"]
labels = ["adaptive-1", "non-adaptive", "random"]
#NAMES = ["ONE-STEP","EXACT","TRACKING","NON-ADAPTIVE","RANDOM"]
NAMES = ["ADAPTIVE","NON-ADAPTIVE","RANDOM"]


name = "../pharmacokinetics/results/opt.txt"
opt = np.loadtxt(name)

ax = plt.axes()
for index, method in enumerate(methods):
    vals = []
    for i in range(10):
        name = "../pharmacokinetics/results/"+ method + "-" + str(i+1)  + ".txt"
        val = np.loadtxt(name)
        vals.append(val)
    median = np.median(opt-np.array(vals),axis = 0)
    q10 = np.quantile(opt-np.array(vals),q=0.1, axis = 0)
    q90 = np.quantile(opt-np.array(vals),q=0.9, axis =0)
    xaxis = np.arange(1,median.shape[0]+1,1)
    plt.plot(xaxis, median, color = color[index], label = NAMES[index])
    plt.fill_between(xaxis, q10, q90, alpha = 0.3, color = color[index])

plt.xlim([2**0, 2**7])
plt.plot(xaxis, 2 ** -19 / np.sqrt(xaxis), 'k', label='1/\u221At')
plt.plot(xaxis, 2 ** -19 / xaxis, 'k:', label='1/t')
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
plt.grid("--", color = 'gray')
plt.xlabel("Episodes [t]")
plt.ylabel("$F(p_t) - F(p^*)$")
plt.xscale('log', base =2 )
plt.yscale('log', base =2 )
plt.legend(fontsize="x-large")
plt.savefig("pharmacokinetics.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)
plt.show()
