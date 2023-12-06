import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

size = (25, 10)
legend_location = 'center left'
fontsize = 20

fig = plt.figure(figsize=size, constrained_layout=False)
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.1)
COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
METHODS = ["adaptive-1","adaptive", "tracking", "non-adaptive", "random"]
#NAMES = ["ONE-STEP","EXACT","TRACKING","NON-ADAPTIVE","RANDOM"]
NAMES = ["LINEAR","EXACT","TRACKING","RESAMPLING","RANDOM"]
NUM_RUNS = 20
LINEWIDTH = [4,4,2,2,2]
def plot(ax, methods, colors, opt_file, base_path, y_lim, label, title):
    opt = np.loadtxt(opt_file)
    for index,method in enumerate(methods):
        vals = []
        for i in range(NUM_RUNS):
            name = base_path + method + "-" + str(i+1)  + ".txt"
            val = np.array(np.loadtxt(name))
            vals.append(val)
        vals = np.array(vals)
        gap = opt - vals
        median = np.median(gap,axis = 0)
        q10 = np.quantile(gap, q=0.1, axis=0)
        q90 = np.quantile(gap, q=0.9, axis=0)
        xaxis = np.arange(1, median.shape[0] + 1, 1)
        ax.plot(xaxis, median, color=colors[index], label=NAMES[index], linewidth = LINEWIDTH[index])
        ax.fill_between(xaxis, q10, q90, alpha=0.3, color=colors[index])
    plt.grid(linestyle = "--", color = 'gray', alpha = 0.5)
    plt.xlabel("Episodes [t]", fontsize=fontsize)
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.yticks(fontsize="x-large")
    plt.xticks(fontsize="x-large")
    plt.ylim(y_lim)
    plt.xlim([2**0, 2**7])
    plt.plot(xaxis, y_lim[-1] / np.sqrt(xaxis), 'k', label='$1/\sqrt{t}$', linewidth = 2)
    plt.plot(xaxis, y_lim[-1] / xaxis, 'k--', label='$1/t$', linewidth = 2)
    plt.plot(xaxis, y_lim[-1] / xaxis**2, 'k-.', label='$1/t^2$', linewidth = 2)

    ax.text(0.0, 1.03, label, transform=ax.transAxes, fontsize=fontsize)
    plt.title(title, fontsize = fontsize)
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.2)

####################################################################################

ax = plt.subplot(gs[0,0])
methods = ["adaptive-1","adaptive","tracking", "density", "random"]
name = "beilschmiedia/results/opt.txt"
plot(ax, methods, COLORS, name, "beilschmiedia/results/", [12,2**9], 'a)', 'Beilschmiedia: variance bound')
plt.ylabel("$F(\eta_t) - F(\eta^*)$", fontsize=fontsize)

####################################################################################

ax = plt.subplot(gs[0,1])
methods = ["adaptive-un-1","adaptive-un","tracking-un", "density-un", "random-un"]
name = "beilschmiedia/results/un-opt.txt"
plot(ax, methods, COLORS, name, "beilschmiedia/results/", [10,2**9], 'b)', 'Beilschmiedia: estimating variance')

####################################################################################

ax = plt.subplot(gs[0,2])
methods = ["adaptive-un-1","adaptive-un", 'tracking-un', "density-un", "random-un"]
name = "pharmacokinetics/results/opt.txt"
plot(ax, methods, COLORS, name, "pharmacokinetics/results/", [10**-8,2**-10], 'c)', 'Pharmacokinetics')

####################################################################################
ax = plt.subplot(gs[0,1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles = handles,
    labels=labels,
    loc=legend_location,
    bbox_to_anchor=(-1.1, -0.225),
    fancybox=False,
    shadow=False,
    ncol=8,
    fontsize=20,
    borderpad=0.1,
    labelspacing=0.1,
    facecolor='grey',
    framealpha=0.2,
)
# plt.title("$\it{Test}$", fontsize = fontsize_legend)
plt.savefig("figs/grids-subplots2.png",dpi = 100, bbox_inches = 'tight',pad_inches = 0)
plt.show()
