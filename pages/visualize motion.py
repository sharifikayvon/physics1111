import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor
from scipy.integrate import cumulative_trapezoid
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.animation as animation
from functools import partial

st.set_page_config(page_title="Visualize 1D Motion", page_icon="🏎️", layout="centered")


def ensure_array(arr, t):
    """
    Ensure arr is a NumPy array with the same length as t.
    If arr is a scalar (int/float), fill an array with that value.
    """
    if np.isscalar(arr):
        return np.full_like(t, arr, dtype=float)
    return np.array(arr, dtype=float)


def makeplot(x_arr, v_arr, a_arr, t, darkmode):

    c1 = "royalblue"
    c2 = "tab:red"
    c3 = "seagreen"
    czero = "k"
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
                "legend.edgecolor": "white",
            }
        )
        c1 = "gold"
        c2 = "lightcyan"
        c3 = "palegreen"
        czero = "gainsboro"

    fig, axes = plt.subplots(
        3, 1, figsize=(8, 8), sharex=True, sharey=False, constrained_layout=True
    )
    axes = axes.flatten()
    axes[2].set_xlabel("Time [s]", fontsize=14)
    ylabels = ["Position [m]", "Velocity [m/s]", r"Acceleration [m/s$^2$]"]
    arrs = [x_arr, v_arr, a_arr]
    cs = [c1, c2, c3]
    for ax, ylabel, arr, c in zip(axes, ylabels, arrs, cs):
        ax.plot(t[10:-10], arr[10:-10], lw=2, color=c, zorder=100)
        ax.set_ylabel(ylabel, fontsize=14)
        # ax.grid(False, which="both")

        ymin, ymax = ax.get_ylim()
        if ymin < 0 < ymax:
            ax.axhline(
                0,
                color=czero,
                lw=2,  # bolder than grid
                alpha=0.8,
                zorder=3,  # above grid, below data
                linestyle="--",
            )
    return fig


def parse_function(func_str):
    """
    Parse a single function string into a sympy expression and a numpy-callable function.

    Returns:
        f: callable, f(t)
        expr: sympy expression
        error: None if successful, else error string
    """
    t_sym = sp.symbols("t")
    transformations = standard_transformations + (convert_xor,)

    func_str = func_str.strip()
    if not func_str:
        return None, None, "Empty input string"

    try:
        expr = parse_expr(
            func_str,
            transformations=transformations,
            local_dict={"t": t_sym, "e": sp.E},
        )
        f = sp.lambdify(t_sym, expr, modules=["numpy"])
        return f, expr, None
    except Exception as e:
        return None, None, f"{func_str} → {e}"


st.title("Visualize 1D Motion 🏎️", text_alignment="center")


delta_t = st.number_input(r"$\Delta t$:", value=3.0, format="%0.3f", step=0.001)


t = np.linspace(0, delta_t, int(1e6))


usr_func = st.radio(
    "Specify function to define:", (r"$x(t)$", r"$v(t)$", r"$a(t)$"), horizontal=True
)
func_str = st.text_area("", "sin(5*t)")

x0 = 0
v0 = 0

t_sym = sp.symbols("t")

if usr_func == r"$x(t)$":

    ufunc, expr, err = parse_function(func_str)

    v_expr = sp.diff(expr, t_sym)
    a_expr = sp.diff(v_expr, t_sym)

    x_func = sp.lambdify(t_sym, expr, modules="numpy")
    v_func = sp.lambdify(t_sym, v_expr, modules="numpy")
    a_func = sp.lambdify(t_sym, a_expr, modules="numpy")

    # x_arr = ufunc(t)
    # v_arr = np.gradient(x_arr, t)
    # a_arr = np.gradient(v_arr, t)


elif usr_func == r"$v(t)$":

    x0 = st.number_input(r"$x_0$:", value=0.0, step=0.001, format="%0.3f")

    ufunc, expr, err = parse_function(func_str)

    x_expr = sp.integrate(expr, t_sym) + x0
    a_expr = sp.diff(expr, t_sym)

    x_func = sp.lambdify(t_sym, x_expr, modules="numpy")
    v_func = sp.lambdify(t_sym, expr, modules="numpy")
    a_func = sp.lambdify(t_sym, a_expr, modules="numpy")
    # v_arr = ufunc(t)
    # a_arr = np.gradient(v_arr, t)
    # x_arr = cumulative_trapezoid(v_arr, t, initial=0.0)
    # x_arr += x0

elif usr_func == r"$a(t)$":

    x0 = st.number_input(r"$x_0$:", value=0.0, step=0.001, format="%0.3f")
    v0 = st.number_input(r"$v_0$:", value=0.0, step=0.001, format="%0.3f")
    ufunc, expr, err = parse_function(func_str)

    v_expr = sp.integrate(expr, t_sym) + v0
    x_expr = sp.integrate(v_expr, t_sym) + x0

    x_func = sp.lambdify(t_sym, x_expr, modules="numpy")
    v_func = sp.lambdify(t_sym, v_expr, modules="numpy")
    a_func = sp.lambdify(t_sym, expr, modules="numpy")

    # a_arr = ufunc(t)
    # # st.write(a_arr.shape)
    # v_arr = cumulative_trapezoid(a_arr, t, initial=0.0)
    # v_arr += v0
    # x_arr = cumulative_trapezoid(v_arr, t, initial=0.0)
    # x_arr += x0


# x_arr = np.array(x_func(t), dtype=float)
# v_arr = np.array(v_func(t), dtype=float)
# a_arr = np.array(a_func(t), dtype=float)

x_arr = ensure_array(x_func(t), t)
v_arr = ensure_array(v_func(t), t)
a_arr = ensure_array(a_func(t), t)


darkmode = st.checkbox("Dark Mode", value=False)

font_path = "static/GoogleSans-Bold.ttf"
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
        "axes.grid": False,
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
        "legend.framealpha": 1.0,
    }
)


fig = makeplot(x_arr, v_arr, a_arr, t, darkmode)


st.pyplot(fig)


buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)

st.download_button(
    label="Download Graph", data=buf, file_name="pva.png", mime="image/png"
)


fig, ax = plt.subplots(figsize=(8, 2))

for xi in range(10):
    arrow = FancyArrowPatch((xi, 0), (xi + 0.8, 0), arrowstyle="->", linewidth=1)
    ax.add_patch(arrow)

ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 0.5)

ax.spines[["left", "right", "top"]].set_visible(False)
ax.set_yticks([])
ax.tick_params(axis="x", bottom=True, labelbottom=True)

st.pyplot(fig)


# fig, ax = plt.subplots()
# t = np.linspace(0, 3, 40)
# g = -9.81
# v0 = 12
# z = g * t**2 / 2 + v0 * t

# v02 = 5
# z2 = g * t**2 / 2 + v02 * t

# scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
# line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
# ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
# ax.legend()


# def update(frame):
#     # for each frame, update the data stored on each artist.
#     x = t[:frame]
#     y = z[:frame]
#     # update the scatter plot:
#     data = np.stack([x, y]).T
#     scat.set_offsets(data)
#     # update the line plot:
#     line2.set_xdata(t[:frame])
#     line2.set_ydata(z2[:frame])
#     return (scat, line2)


# ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
# plt.show()


fig, ax = plt.subplots()
x = np.linspace(0, 2 * np.pi, 200)
(line,) = ax.plot(x, np.sin(x))


def update(frame):
    line.set_ydata(np.sin(x + frame / 10))
    return (line,)


ani = FuncAnimation(fig, update, frames=100, interval=50)

# html = ani.to_jshtml()
# st.components.v1.html(html, height=400)

# fig, ax = plt.subplots()
# line1, = ax.plot([], [], 'ro')

# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     return line1,

# def update(frame, ln, x, y):
#     x.append(frame)
#     y.append(np.sin(frame))
#     ln.set_data(x, y)
#     return ln,

# ani = FuncAnimation(
#     fig, partial(update, ln=line1, x=[], y=[]),
#     frames=np.linspace(0, 2*np.pi, 128),
#     init_func=init, blit=True)

# while 

# st.pyplot(fig)
# ani.save("animation.gif", writer=PillowWriter(fps=20))
# plt.close(fig)

# st.image("animation.gif")


# fig, ax = plt.subplots()
# t = np.linspace(0, 3, 40)
# g = -9.81
# v0 = 12
# z = g * t**2 / 2 + v0 * t

# v02 = 5
# z2 = g * t**2 / 2 + v02 * t

# scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
# line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
# ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
# ax.legend()


# def update(frame):
#     # for each frame, update the data stored on each artist.
#     x = t[:frame]
#     y = z[:frame]
#     # update the scatter plot:
#     data = np.stack([x, y]).T
#     scat.set_offsets(data)
#     # update the line plot:
#     line2.set_xdata(t[:frame])
#     line2.set_ydata(z2[:frame])
#     return (scat, line2)


# ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)

# ani = FuncAnimation(fig, update, frames=100)


# st.write(animation.writers.list())
ani.save("anim.mp4", writer=FFMpegWriter(fps=10))
st.video("anim.mp4")
st.markdown('at the bottom!')