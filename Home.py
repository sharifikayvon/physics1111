import streamlit as st

st.set_page_config(
    page_title="phys1111toolkit",
    page_icon="🍎",
    layout="wide"
)
st.title("phys1111 Data Visualization and Analysis Toolkit", text_alignment="center")

st.markdown("""
Welcome to the **phys1111 Data Analysis Toolkit**! This website is designed to 
streamline routine in-class data visualization and analysis tasks for students in 
Dr. Barooni's Physics 1111 course and to provide interactive demos of course concepts to aid students' understanding.

Features include:
- **Graph and Fit your Data**: Make a scatter plot of your data and fit a linear or quadratic function.
- **Visualize 1D Motion**: Define **x(t)**, **v(t)**, or **a(t)** and visualize all three simultaneously.

Explore the tools using the sidebar on the left. If you have any questions or feedback, please email me at ksharifi1@gsu.edu.

More features to be added soon.


""")