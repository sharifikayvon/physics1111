import streamlit as st
import sympy as sp
import numpy as np
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor
)

def parse_functions(multiline_str):
    x = sp.symbols('x')

    functions = []
    expressions = []
    errors = []

    # Enable ^ → ** conversion
    transformations = standard_transformations + (convert_xor,)

    lines = multiline_str.strip().splitlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            # Parse with fixes:
            expr = parse_expr(
                line,
                transformations=transformations,
                local_dict={"x": x, "e": sp.E}
            )

            f = sp.lambdify(x, expr, modules=["numpy"])

            functions.append(f)
            expressions.append(expr)

        except Exception as e:
            errors.append(f"Line {i}: {line} → {e}")

    return functions, expressions, errors

# def parse_functions(multiline_str):
#     x = sp.symbols('x')

#     functions = []
#     expressions = []

#     # Split input into lines
#     lines = multiline_str.strip().splitlines()

#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue  # skip empty lines

#         # Parse user input into a sympy expression
#         expr = sp.sympify(line)

#         # Convert to a numpy-aware function
#         f = sp.lambdify(x, expr, modules=["numpy"])

#         functions.append(f)
#         expressions.append(expr)

#     return functions, expressions


st.title("Multi-Function Plotter")

expr_str = st.text_area(
    "Enter one function per line (in terms of x):",
    value="5*x^2\nsin(x)\ne^x"
)

if expr_str:
    try:
        funcs, exprs, errors = parse_functions(expr_str)

        x_vals = np.linspace(-10, 10, 400)

        for f, expr in zip(funcs, exprs):
            y_vals = f(x_vals)
            st.line_chart({"x": x_vals, str(expr): y_vals})

    except Exception as e:
        st.error(f"Invalid function: {e}")


st.write(exprs)
# st.title("Plot a Function", text_alignment="center")


# eq_str = st.text_area("Enter a function of x", "sin(x) * exp(-0.1*x)")




