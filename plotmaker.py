import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

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

def makeplot(xdata, ydata, xlabel='x axis', ylabel='y axis', title='Title', fitline=False, fitquad=False, savefig=False):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.scatter(xdata,ydata, s=10, c='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both')

    if fitline:
        lin_coeffs = np.polyfit(xdata, ydata, 1)
        lin_xfit = np.linspace(np.min(xdata), np.max(xdata), 500)
        lin_yfit = np.polyval(lin_coeffs, lin_xfit)
        lin_label = f'Linear Fit: y = {lin_coeffs[0]:.3g}x + {lin_coeffs[1]:.3g}' 
        ax.plot(lin_xfit, lin_yfit, color='r', linestyle='--', label=lin_label)
        ax.legend()
    
    if fitquad:      
        quad_coeffs = np.polyfit(xdata, ydata, 2)
        quad_xfit = np.linspace(np.min(xdata), np.max(xdata), 500)
        quad_yfit = np.polyval(quad_coeffs, quad_xfit)
        quad_label = f'Quadratic Fit: y = {quad_coeffs[0]:.3g}x² + {quad_coeffs[1]:.3g}x + {quad_coeffs[2]:.3g}'
        ax.plot(quad_xfit, quad_yfit, color='b', linestyle='--', label=quad_label)
        ax.legend()

    if savefig:
        outfile = title.replace(" ", "_") + ".png"
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
    
    plt.show()
