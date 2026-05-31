# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.8",
#     "matplotlib>=3.10.9",
#     "numpy>=2.4.6",
# ]
# [tool.uv]
# exclude-newer = "2026-06-01T00:00:00Z"
# ///
import marimo

__generated_with = "0.23.8"
app = marimo.App(
    width="medium",
    layout_file="layouts/interactive_slides.slides.json",
)


@app.cell
def _():
    # Hints:
    #  - To run this notebook: uv run marimo run interactve_slides.py
    #  - To edit:              uv run marimo edit interactve_slides.py
    #                          and then press CTRL/CMD+. to toggle slides
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    # Large fonts, etc.
    plt.style.use("seaborn-v0_8-poster")
    return mo, np, plt


@app.cell
def _(mo):
    w1 = mo.ui.slider(-2, 2, step=0.01, value=-0.3, full_width=True)
    w2 = mo.ui.slider(-2, 2, step=0.01, value=0.3, full_width=True)
    b = mo.ui.slider(-2, 2, step=0.01, value=-0.2, full_width=True)
    return b, w1, w2


@app.cell
def _(b, mo, np, plt, w1, w2):
    decision_boundary_sliders = mo.md(f"""
    $w_1$ = {w1.value}
    {w1}

    $w_2$ = {w2.value}
    {w2}

    $b$ = {b.value}
    {b}
    """)


    def plot_and_gate():
        fig, ax = plt.subplots(figsize=(8, 6))

        # Decision boundary: w1*x1 + w2*x2 + b = 0
        x1, x2 = np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100))
        y = w1.value * x1 + w2.value * x2 + b.value
        ax.contour(x1, x2, y, levels=[0], colors="black", linewidths=2)
        ax.contourf(x1, x2, y, levels=[-np.inf, 0], colors="gray", alpha=0.3)

        # Data points
        plt.scatter(
            [0, 1, 0],
            [1, 0, 0],
            color="blue",
            marker="o",
            s=100,
            label="Class 0",
        )
        plt.scatter(
            [1],
            [1],
            color="red",
            marker="^",
            s=100,
            label="Class 1",
        )

        # Equation
        if w2.value != 0:
            slope = -w1.value / w2.value
            intercept = -b.value / w2.value
            equation_text = f"$x_2 = {slope:.2f}x_1 + {intercept:.2f}$"
        else:
            equation_text = f"$x_1 = {-b.value / w1.value:.2f}$"

        # Plot settings
        ax.set(
            title=f"Decision Boundary: {equation_text}",
            xlim=(-0.5, 1.5),
            ylim=(-0.5, 1.5),
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="$x_1$",
            ylabel="$x_2$",
        )
        ax.legend()

        return ax


    and_gate_graph = plot_and_gate()
    return and_gate_graph, decision_boundary_sliders


@app.cell
def _(and_gate_graph, decision_boundary_sliders, mo):
    mo.hstack(
        [
            mo.vstack([mo.md("# AND gate"), decision_boundary_sliders]),
            and_gate_graph,
        ]
    )
    return


@app.cell
def _(b, np, plt, w1, w2):
    def plot_xor_gate():
        fig, ax = plt.subplots(figsize=(8, 6))

        # Decision boundary: w1*x1 + w2*x2 + b = 0
        x1, x2 = np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100))
        y = w1.value * x1 + w2.value * x2 + b.value
        ax.contour(x1, x2, y, levels=[0], colors="black", linewidths=2)
        ax.contourf(x1, x2, y, levels=[-np.inf, 0], colors="gray", alpha=0.3)

        # Data points
        plt.scatter(
            [0, 1],
            [0, 1],
            color="blue",
            marker="o",
            s=100,
            label="Class 0",
        )
        plt.scatter(
            [1, 0],
            [0, 1],
            color="red",
            marker="^",
            s=100,
            label="Class 1",
        )

        # Equation
        if w2.value != 0:
            slope = -w1.value / w2.value
            intercept = -b.value / w2.value
            equation_text = f"$x_2 = {slope:.2f}x_1 + {intercept:.2f}$"
        else:
            equation_text = f"$x_1 = {-b.value / w1.value:.2f}$"

        # Plot settings
        ax.set(
            title=f"Decision Boundary: {equation_text}",
            xlim=(-0.5, 1.5),
            ylim=(-0.5, 1.5),
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="$x_1$",
            ylabel="$x_2$",
        )
        ax.legend()

        return ax


    xor_gate_graph = plot_xor_gate()
    return (xor_gate_graph,)


@app.cell
def _(decision_boundary_sliders, mo, xor_gate_graph):
    mo.hstack(
        [
            mo.vstack([mo.md("# XOR gate"), decision_boundary_sliders]),
            xor_gate_graph,
        ]
    )
    return


@app.cell
def _(mo, np):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def relu(x):
        return np.maximum(0, x)


    def tanh(x):
        return np.tanh(x)


    W_B = mo.ui.matrix(
        [[2, 0.8, 0.8], [1.2, 1, -1], [0.5, -2.5, -0.1]],
        step=0.1,
        column_labels=["x1", "x2", "b"],
        row_labels=["neuron 1", "neuron 2", "neuron 3"],
    )
    activation = mo.ui.dropdown(
        {"sigmoid": sigmoid, "relu": relu, "tanh": tanh},
        value="relu",
    )
    return W_B, activation


@app.cell
def _(W_B, activation, mo, np, plt):
    activation_sliders = mo.md(f"""
    Weights and biases {W_B}

    Activation function {activation}
    """)


    def plot_activation():
        fig, ax = plt.subplots(figsize=(8, 6))

        # Decision boundary
        x1, x2 = np.meshgrid(np.linspace(-1, 2, 50), np.linspace(-1, 2, 50))
        (
            [neuron1_w1, neuron1_w2, neuron1_b],
            [neuron2_w1, neuron2_w2, neuron2_b],
            [neuron3_w1, neuron3_w2, neuron3_b],
        ) = W_B.value
        neuron1_output = activation.value(
            neuron1_w1 * x1 + neuron1_w2 * x2 + neuron1_b
        )
        neuron2_output = activation.value(
            neuron2_w1 * x1 + neuron2_w2 * x2 + neuron2_b
        )
        y = neuron1_output * neuron3_w1 + neuron2_output * neuron3_w2 + neuron3_b
        ax.contour(x1, x2, y, levels=[0.5], colors="black", linewidths=2)
        ax.contourf(x1, x2, y, levels=[-np.inf, 0.5], colors="gray", alpha=0.3)

        # Data points
        plt.scatter(
            [0, 1],
            [0, 1],
            color="blue",
            marker="o",
            s=100,
            label="Class 0",
        )
        plt.scatter(
            [1, 0],
            [0, 1],
            color="red",
            marker="^",
            s=100,
            label="Class 1",
        )

        # Plot settings
        ax.set(
            xlim=(-0.5, 1.5),
            ylim=(-0.5, 1.5),
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="$x_1$",
            ylabel="$x_2$",
        )
        ax.legend()

        return ax


    activation_function_graph = plot_activation()
    return activation_function_graph, activation_sliders


@app.cell
def _(activation_function_graph, activation_sliders, mo):
    mo.hstack(
        [
            mo.vstack(
                [mo.md("# XOR gate with activation function"), activation_sliders]
            ),
            activation_function_graph,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
