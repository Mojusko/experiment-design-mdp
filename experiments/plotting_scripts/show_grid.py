import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data

from mdpexplore.env.grid_worlds import DummyGridWorld

GRID_WIDTH, GRID_HEIGHT = 7, 6
TRAJECTORY_FILES = [ f"test{i}.txt" for i in range(10) ]
TRAJECTORIES = [ np.loadtxt(f).astype(int) for f in TRAJECTORY_FILES ]
NUM_TRAJECTORIES = 3



def plot_sectors(sectormap_np, ax):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            image_path = "../../resources/" + str(int(sectormap_np[y,x])) + ".png"
            imscatter(x+0.5, y+0.5, image_path, zoom=0.25, ax=ax)
            ax.plot(x+0.5, y+0.5)

def imscatter(x, y, image, ax, zoom=1):
    image = plt.imread(image)
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

if __name__ == "__main__" :

    heatmap = {}
    for t in TRAJECTORIES:
        for i in range(t.shape[0]):
            heatmap[(t[i,0], t[i,1])] = heatmap.get((t[i,0], t[i,1]), 0) + 1

    peak = heatmap[max(heatmap, key=heatmap.get)]
    heatmap = {k: v/peak for k, v in heatmap.items()}

    dummy = DummyGridWorld()
    sector_list = dummy._generate_sector_ids(sectors_num=None)
    sector_list = [[dummy.convert_to_grid(s) for s in l] for l in sector_list]
    sectormap = {}
    for i, l in enumerate(sector_list):
        for s in l:
            sectormap[s] = i
            if s not in heatmap:
                heatmap[s] = 0

    heatmap_np, sectormap_np = np.zeros((GRID_WIDTH, GRID_HEIGHT)), np.zeros((GRID_WIDTH, GRID_HEIGHT))
    for s in sectormap:
        heatmap_np[s] = heatmap[s]
        sectormap_np[s] = sectormap[s]

    heatmap_np = heatmap_np[:, ::-1].T
    sectormap_np = sectormap_np[:, ::-1].T

    fig = plt.figure(figsize=(15,15))
    ax = plt.subplot(1,1,1, aspect='equal')

    sb.heatmap(
        heatmap_np,
        cmap="Reds",
        ax=ax,
        linewidths=1,
        alpha=0.9,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )

    plot_sectors(sectormap_np, ax)

    plt.savefig("heatmap.png", dpi = 100, bbox_inches = 'tight',pad_inches = 0)
    plt.clf()

    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1, aspect='equal')
    plot_sectors(sectormap_np, ax)
    plt.savefig("trajs_base.png", dpi=100, bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1, aspect='equal')
    heatmap = {k: 0 for k, v in heatmap.items()}
    for j, t in enumerate(TRAJECTORIES[0:NUM_TRAJECTORIES]):
        for i in range(t.shape[0]):
            heatmap[(t[i,0], t[i,1])] = j + 1 if heatmap[(t[i,0], t[i,1])] in [0, j+1] else NUM_TRAJECTORIES+1
        heatmap_np = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        for s in heatmap:
            heatmap_np[s] = heatmap[s]
        heatmap_np = heatmap_np[:, ::-1].T

        sb.heatmap(
            heatmap_np,
            cmap=['white', 'lightcoral', 'lightgreen', 'lightblue', 'darkslateblue'],
            ax=ax,
            linewidths=1,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            vmin=0,
            vmax=4,
        )

        plot_sectors(sectormap_np, ax)

        plt.savefig("trajs"+str(j)+".png", dpi = 100, bbox_inches = 'tight',pad_inches = 0)
        #plt.clf()
