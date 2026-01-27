import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd

st.set_page_config(
    page_title="Visualize 1D Motion",
    page_icon="🏎️",
    layout="centered"
)


# title = st.text_input("Plot Title:", "Title")

# col1, col2 = st.columns(2)
# xlabel = col1.text_input("X Label:", "x axis")
# ylabel = col2.text_input("Y Label:", "y axis")


col1, col2, col3 = st.columns(3)
fitline = col1.checkbox("Linear Fit", value=False)
fitquad = col2.checkbox("Quadratic Fit", value=False)
darkmode = col3.checkbox("Dark Mode", value=False)

font_path = 'static/Barlow-Regular.ttf'
mpl.font_manager.fontManager.addfont(font_path)
font_prop = mpl.font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

mpl.rcParams.update(
    {
            "figure.dpi": 200,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "savefig.dpi": 300,
            "savefig.format": "png",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white", 
            "savefig.edgecolor": "white",
            "figure.autolayout": True,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.labelcolor": "black",
            "axes.labelsize": 20,
            "axes.titlesize": 20,
            "axes.titlecolor": "black",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.color": "black",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.8,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,
            "xtick.color": "black", 
            "ytick.color": "black",
            "xtick.labelcolor": "black", 
            "ytick.labelcolor": "black",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.minor.ndivs": 5,
            "ytick.minor.ndivs": 5,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "lines.color": "black",
            "mathtext.default": "regular",
            "legend.frameon": True,
            "legend.fontsize": 12,
            "legend.handlelength": 2,
            "legend.labelcolor": "black",
            "legend.facecolor": "white",
            "legend.edgecolor": "black",
            "legend.fancybox": True,
            "legend.framealpha": 1.0
    }
)

c = 'k'
edgecolors = 'gainsboro'
c1 = 'dodgerblue'
c2 = 'orangered'

if darkmode:
    mpl.rcParams.update(
        {
            "figure.facecolor": "black",
            "figure.edgecolor": "black",
            "savefig.facecolor": "black",
            "savefig.edgecolor": "black",
            "axes.facecolor": "black",
            "axes.edgecolor": "white",
            "axes.labelcolor": "white",
            "axes.titlecolor": "white",
            "grid.color": "snow",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.8,
            "xtick.color": "white",
            "ytick.color": "white",
            "xtick.labelcolor": "white",
            "ytick.labelcolor": "white",
            "lines.color": "white",
            "mathtext.default": "regular",
            "legend.labelcolor": "white",
            "legend.facecolor": "black",
            "legend.edgecolor": "white"
        }
    )
    c = 'gainsboro'
    edgecolors = 'w'
    c1 = 'lime'
    c2 = "cyan"


fig, axes = plt.subplots(3,1,figsize=(8, 8), sharex=True, sharey=True, constrained_layout=True)
axes = axes.flatten()
axes[2].set_xlabel('Time', fontsize=14)

ylabels = ['Position [m]', 'Velocity [m/s]', r'Acceleration [m/s$^2$]' ]

for ax, ylabel in zip(axes, ylabels):
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, which="both")



st.pyplot(fig)



buf = BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)

st.download_button(
    label="Download Graph",
    data=buf,
    file_name="pva.png",
    mime="image/png"
)