import streamlit as st
from pathlib import Path
import base64

st.set_page_config(
    page_title="phys1111tools",
    page_icon="🍎",
    layout="wide"
)

font_path = Path(__file__).parent / "Barlow-Regular.ttf"
font_base64 = base64.b64encode(font_path.read_bytes()).decode()


st.markdown(
    f"""
    <style>
    @font-face {{
        font-family: 'Barlow';
        src: url('{font_path.as_posix()}') format('truetype');
        font-weight: normal;
        font-style: normal;
    }}

    html, body, [class*="css"]  {{
        font-family: 'Barlow', sans-serif;
    }}
    </style>
    """,
    unsafe_allow_html=True
)






st.title("PHYS 1111 Data Analysis Tools hello", text_alignment="center")


st.markdown("""
Welcome to the **PHYS 1111 Data Analysis Tools** app! This website was designed to streamline students' daily data visualization and analysis tasks for Dr. Barooni's PHYS 1111 course.
- **Calculate Uncertainty**: Calculate uncertainty from repeated measurements and visualize the distribution.
- **Graph and Fit your Data**: Make a scatter plot of your data and fit a linear or quadratic function.
- **Plot Functions**: Visualize functions over a specified range.

Explore the tools using the sidebar on the left! If you have any questions or feedback, please email me at ksharifi1@gsu.edu.

More features to be added soon. Stay tuned!


""")