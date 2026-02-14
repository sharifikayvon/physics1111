import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import streamlit as st
from io import BytesIO
import pandas as pd

st.set_page_config(
    page_title="Graph and Fit your Data", page_icon="📈", layout="centered"
)


def has_valid_xy(x, y):
    return (
        x is not None
        and y is not None
        and len(x) >= 2
        and len(y) >= 2
        and np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
    )


def fmt_term(coef, term="", sig=3, tol=1e-8, first=False):
    """Format a single polynomial term with proper signs, parentheses, and skipping 1 before x^n."""
    if abs(coef) < tol:
        return ""

    # Number formatting
    def _fmt(x):
        if x == 0:
            return "0"
        # Use scientific notation if very small or very large
        if abs(x) < 1e-4 or abs(x) > 1e4:
            base, exp = f"{x:.{sig}e}".split("e")
            return rf"({base}\times 10^{{{int(exp)}}})"
        return f"{x:.{sig}g}"

    mag = _fmt(abs(coef))

    if term and mag == "1":
        mag = ""

    if term and (mag.startswith("(") or "e" in mag):
        mag = f"({mag})"

    body = rf"{mag}{term}"

    if first:
        return rf"- {body}" if coef < 0 else body
    else:
        sign = "-" if coef < 0 else "+"
        return rf"{sign} {body}"


def fmt_poly(coeffs, terms, sig=3, tol=1e-8):
    """
    Format a polynomial from lists of coefficients and term strings.

    coeffs : list or array of coefficients [a_n, a_{n-1}, ..., a0]
    terms  : list of term strings ['x^2', 'x', ''] etc.
    """
    # find first non-zero coefficient
    first_idx = next((i for i, c in enumerate(coeffs) if abs(c) >= tol), None)
    if first_idx is None:
        return "0"

    parts = []
    for i, (c, t) in enumerate(zip(coeffs, terms)):
        is_first = i == first_idx
        parts.append(fmt_term(c, t, sig=sig, tol=tol, first=is_first))

    return "".join(parts)


st.markdown("<h1 style='text-align: center'>Graph and Fit your Data 📈</h1>", unsafe_allow_html=True)

mode = st.radio(
    "Choose data input method:",
    ("Manually enter data", "Upload data file"),
    horizontal=True,
)


if mode == "Upload data file":
    uploaded_file = st.file_uploader(
        "Upload a data file (.csv, .txt, .xlsx)", type=["csv", "txt", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep="\s+")
        else:
            df = pd.read_csv(uploaded_file, sep=",")
        preview = st.checkbox("Preview Data?", value=False)
        # st.write("Preview:")
        if preview:
            st.dataframe(df.head(), hide_index=True)

        describe = st.checkbox("Describe Data?", value=False)
        if describe:
            st.dataframe(df.describe())

        x_col = st.selectbox("X column", df.columns)
        y_col = st.selectbox("Y column", df.columns)

        xdata = df[x_col].values
        ydata = df[y_col].values

elif mode == "Manually enter data":
    st.markdown("Enter values separated by spaces or commas.")

    x_str = st.text_area("X values", "0 1 2 3 4 5")
    y_str = st.text_area("Y values", "0 1 4 9 16 25")

    def parse_array(s):
        return np.array([float(v) for v in s.replace(",", " ").split()])

    try:
        xdata = parse_array(x_str)
        ydata = parse_array(y_str)

        if len(xdata) != len(ydata):
            st.error("X and Y must have the same length.")
            st.stop()

    except ValueError:
        st.error("Could not parse numbers. Please check your input.")
        st.stop()


title = st.text_input("Plot Title:", "Title")

col1, col2 = st.columns(2)
xlabel = col1.text_input("X Label:", "x axis")
ylabel = col2.text_input("Y Label:", "y axis")


col1, col2, col3 = st.columns(3)
fitline = col1.checkbox("Linear Fit", value=False)
fitquad = col2.checkbox("Quadratic Fit", value=False)
darkmode = col3.checkbox("Dark Mode", value=False)
force_origin = False
if fitline:
    force_origin = st.checkbox("Set Linear Fit y-intercept to 0", value=False)

font_path = "static/GoogleSans-Regular.ttf"
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
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
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

c = "k"
edgecolors = "gainsboro"
c1 = "dodgerblue"
c2 = "orangered"

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
    c = "gainsboro"
    edgecolors = "w"
    c1 = "lime"
    c2 = "cyan"

if "xdata" not in locals() and "ydata" not in locals():
    xdata = np.arange(-3 * np.pi, 3 * np.pi, 0.05)
    ydata = np.sin(-0.5 * xdata) + np.random.normal(0, 0.15, len(xdata))
    # xdata = np.arange(0,10, 1)
    # ydata = 2 * xdata**2 + 5*xdata + np.random.normal(0, 3, len(xdata))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(xdata, ydata, s=100, c=c, edgecolors=edgecolors, lw=3)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title(title)
ax.grid(True, which="both")

# if fitline and has_valid_xy(xdata, ydata):
#     lin_coeffs = np.polyfit(xdata, ydata, 1)
#     lin_xfit = np.linspace(np.min(xdata), np.max(xdata), 500)
#     lin_yfit = np.polyval(lin_coeffs, lin_xfit)
#     # lin_label = rf"$Linear\ Fit:\ y\ =\ {fmt(lin_coeffs[0])}x\ +\ {fmt(lin_coeffs[1])}$"
#     # lin_label = rf"$Linear\ Fit:\ y\ =\ {fmt_term(lin_coeffs[0], 'x', first=True)}{fmt_term(lin_coeffs[1])}$"
#     lin_label = rf"$Linear\ Fit:\ y = {fmt_poly(lin_coeffs, ['x',''])}$"
#     ax.plot(lin_xfit, lin_yfit, color=c1, linestyle="solid", label=lin_label, lw=3)
#     ax.legend()
if fitline and has_valid_xy(xdata, ydata):

    if force_origin:
        # constrained fit through (0,0)
        m = np.dot(xdata, ydata) / np.dot(xdata, xdata)

        x_min = np.min(xdata)
        x_max = np.max(xdata)
        lin_xfit = np.array([x_min, x_max])
        lin_yfit = m * lin_xfit

        lin_label = rf"$Linear\ Fit:\ y = {fmt_term(m, 'x', first=True)}$"

    else:
        lin_coeffs = np.polyfit(xdata, ydata, 1)
        lin_xfit = np.linspace(np.min(xdata), np.max(xdata), 500)
        lin_yfit = np.polyval(lin_coeffs, lin_xfit)

        lin_label = rf"$Linear\ Fit:\ y = {fmt_poly(lin_coeffs, ['x',''])}$"

    ax.plot(lin_xfit, lin_yfit, color=c1, linestyle="solid", label=lin_label, lw=3)
    ax.legend()
elif fitline:
    st.warning("Cannot fit line: data is empty or invalid")

if fitquad and has_valid_xy(xdata, ydata):
    quad_coeffs = np.polyfit(xdata, ydata, 2)
    quad_xfit = np.linspace(np.min(xdata), np.max(xdata), 500)
    quad_yfit = np.polyval(quad_coeffs, quad_xfit)
    # quad_label = rf"$Quadratic\ Fit:\ y\ =\ {fmt(quad_coeffs[0])}x^2\ +\ {fmt(quad_coeffs[1])}x\ +\ {fmt(quad_coeffs[2])}$"
    # quad_label = rf"$Quadratic\ Fit:\ y\ =\ {fmt_term(quad_coeffs[0], 'x^2', first=True)}{fmt_term(quad_coeffs[1], 'x')}{fmt_term(quad_coeffs[2])}$"
    quad_label = rf"$Quadratic\ Fit:\ y = {fmt_poly(quad_coeffs, ['x^2','x',''])}$"
    ax.plot(quad_xfit, quad_yfit, color=c2, linestyle="dashed", label=quad_label, lw=3)
    ax.legend()
elif fitquad:
    st.warning("Cannot fit parabola: data is empty or invalid")


st.pyplot(fig)

buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)

st.download_button(
    label="Download Graph",
    data=buf,
    file_name=f"{title.replace(' ', '_')}.png",
    mime="image/png",
)
