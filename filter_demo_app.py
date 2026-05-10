# /// script
# dependencies = [
#   "marimo",
#   "matplotlib",
#   "numpy",
#   "pandas",
# ]
# ///

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from filter_demo_core import run_demo

    return Ellipse, mo, np, pd, plt, run_demo


@app.cell
def _(mo):
    mo.md(
        """
        # 2D Filter Comparison for Robotics

        This demo compares three related ideas on the same quadcopter-style mission from point **A** to point **B**:

        - **Markov approach**: propagate belief using only the motion model
        - **Bayesian filter**: motion prediction plus sensor correction on a discrete 2D grid
        - **Kalman filter**: continuous Gaussian estimate with mean and covariance

        The point is not that one method is always “best”, but that they represent uncertainty in different ways.
        """
    )
    return


@app.cell
def _(mo):
    steps = mo.ui.slider(12, 40, value=24, step=1, label="Number of time steps")
    grid_size = mo.ui.dropdown(
        options={"21": 21, "25": 25, "31": 31},
        value=25,
        label="Grid resolution",
    )
    motion_sigma = mo.ui.slider(0.15, 1.2, value=0.55, step=0.05, label="Motion noise")
    measurement_sigma = mo.ui.slider(0.15, 1.5, value=0.7, step=0.05, label="Measurement noise")
    prior_sigma = mo.ui.slider(0.25, 1.8, value=0.95, step=0.05, label="Initial uncertainty")
    measurement_mode = mo.ui.dropdown(
        options={"x-only sensor": "x-only", "full 2D sensor": "xy"},
        value="x-only",
        label="Measurement model",
    )
    seed = mo.ui.slider(0, 50, value=7, step=1, label="Random seed")
    controls = mo.vstack(
        [
            mo.hstack([steps, grid_size, measurement_mode]),
            mo.hstack([motion_sigma, measurement_sigma, prior_sigma, seed]),
        ]
    )
    return (
        controls,
        grid_size,
        measurement_mode,
        measurement_sigma,
        motion_sigma,
        prior_sigma,
        seed,
        steps,
    )


@app.cell
def _(controls, mo):
    mo.md("## Controls")
    controls
    return


@app.cell
def _(grid_size, measurement_mode, measurement_sigma, motion_sigma, prior_sigma, run_demo, seed, steps):
    results = run_demo(
        steps=steps.value,
        grid_size=grid_size.value,
        motion_sigma=motion_sigma.value,
        measurement_sigma=measurement_sigma.value,
        prior_sigma=prior_sigma.value,
        measurement_mode=measurement_mode.value,
        seed=seed.value,
    )
    return (results,)


@app.cell
def _(mo, results):
    step = mo.ui.slider(
        0,
        len(results.true_positions) - 1,
        value=min(8, len(results.true_positions) - 1),
        step=1,
        label="Inspect time step",
    )
    step
    return (step,)


@app.cell
def _(Ellipse, np, plt, results, step):
    idx = step.value
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    titles = [
        "Markov propagation only",
        "Bayesian histogram filter",
        "Kalman filter",
    ]
    beliefs = [results.markov_beliefs[idx], results.bayes_beliefs[idx]]
    means = [results.markov_means[idx], results.bayes_means[idx], results.kalman_means[idx]]
    truth = results.true_positions
    true_xy = truth[idx]
    if results.measurement_mode == "x-only":
        meas_xy = np.array([results.measurements[idx, 0], true_xy[1]])
        meas_label = "x-only measurement"
    else:
        meas_xy = results.measurements[idx]
        meas_label = "2D measurement"

    for ax, title, belief, mean in zip(axes[:2], titles[:2], beliefs, means[:2]):
        ax.imshow(
            belief,
            origin="lower",
            extent=[results.xs[0], results.xs[-1], results.ys[0], results.ys[-1]],
            cmap="viridis",
            aspect="equal",
        )
        ax.plot(truth[:, 0], truth[:, 1], "--", color="white", alpha=0.75, linewidth=1.5)
        ax.scatter(*true_xy, color="red", s=65, label="true state")
        ax.scatter(*mean, color="cyan", s=55, label="belief mean")
        ax.scatter(*meas_xy, color="orange", s=40, label=meas_label)
        ax.set_title(title)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax = axes[2]
    ax.plot(truth[:, 0], truth[:, 1], "--", color="#4A6FA5", alpha=0.6, linewidth=1.5)
    ax.scatter(*true_xy, color="red", s=65, label="true state")
    ax.scatter(*results.kalman_means[idx], color="#433AB5", s=60, label="KF mean")
    ax.scatter(*meas_xy, color="orange", s=40, label=meas_label)
    cov = results.kalman_covs[idx]
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(np.maximum(vals, 1e-9)) * 2.0
    ellipse = Ellipse(
        xy=results.kalman_means[idx],
        width=width,
        height=height,
        angle=angle,
        edgecolor="#433AB5",
        facecolor="#433AB522",
        linewidth=2,
    )
    ax.add_patch(ellipse)
    ax.set_title(titles[2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    axes[0].legend(loc="upper right", fontsize=8)
    return fig


@app.cell
def _(plt, results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
    t = range(len(results.true_positions))

    axes[0].plot(t, results.markov_errors, label="Markov only", linewidth=2)
    axes[0].plot(t, results.bayes_errors, label="Bayes filter", linewidth=2)
    axes[0].plot(t, results.kalman_errors, label="Kalman filter", linewidth=2)
    axes[0].set_title("Position error over time")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("Euclidean error")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(t, results.markov_entropy, label="Markov entropy", linewidth=2)
    axes[1].plot(t, results.bayes_entropy, label="Bayes entropy", linewidth=2)
    axes[1].plot(t, results.kalman_trace, label="KF covariance trace", linewidth=2)
    axes[1].set_title("Uncertainty trend")
    axes[1].set_xlabel("time step")
    axes[1].set_ylabel("uncertainty metric")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    return fig


@app.cell
def _(mo, results):
    markov_final = results.markov_errors[-1]
    bayes_final = results.bayes_errors[-1]
    kalman_final = results.kalman_errors[-1]
    mode_text = (
        "The x-only sensor makes the uncertainty stripe-like in 2D, which helps students see why the motion model matters."
        if results.measurement_mode == "x-only"
        else "The full 2D sensor makes the Kalman and Bayes filters converge faster, while Markov-only still drifts."
    )
    mo.md(
        f"""
        ## How to read the comparison

        - **Markov approach**: this keeps only the motion-model propagation. Its belief usually spreads, because there is no correction step.
        - **Bayesian filter**: this is the full discrete-grid correction story. It can represent broad or even non-Gaussian beliefs.
        - **Kalman filter**: this compresses belief into a mean and covariance ellipse, which is efficient when a Gaussian approximation is reasonable.

        **Current run summary**

        - Final Markov-only error: `{markov_final:.3f}`
        - Final Bayes-filter error: `{bayes_final:.3f}`
        - Final Kalman-filter error: `{kalman_final:.3f}`

        {mode_text}
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Teaching suggestions

        1. Start with the **Markov panel** and tell students this is what happens if we only trust motion.
        2. Move to the **Bayes panel** and explain how measurements reshape the belief.
        3. Finish with the **Kalman panel** and emphasize the Gaussian assumption: compact, fast, but less expressive than a full grid belief.
        """
    )
    return


if __name__ == "__main__":
    app.run()
