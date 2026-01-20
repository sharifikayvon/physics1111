import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "xtick.top": True,
    "ytick.right": True,
    "xtick.bottom": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "xtick.minor.width": 1.5,
    "ytick.minor.width": 1.5,
    "xtick.minor.ndivs": 5,
    "ytick.minor.ndivs": 5,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.color": "k",
    "grid.linewidth": 0.75,
})

def makeplot(xdata, ydata, xlabel='x axis', ylabel='y axis', title='Title'):

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(xdata,ydata, marker='o', color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both')
    plt.show()
