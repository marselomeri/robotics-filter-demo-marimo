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

    try:
        from filter_demo_core import run_demo
    except ModuleNotFoundError:
        from dataclasses import dataclass

        @dataclass
        class DemoResults:
            xs: np.ndarray
            ys: np.ndarray
            true_positions: np.ndarray
            measurements: np.ndarray
            measurement_mode: str
            markov_beliefs: np.ndarray
            bayes_beliefs: np.ndarray
            markov_means: np.ndarray
            bayes_means: np.ndarray
            kalman_means: np.ndarray
            kalman_covs: np.ndarray
            markov_errors: np.ndarray
            bayes_errors: np.ndarray
            kalman_errors: np.ndarray
            markov_entropy: np.ndarray
            bayes_entropy: np.ndarray
            kalman_trace: np.ndarray

        def build_trajectory(
            steps: int,
            start: tuple[float, float] = (1.0, 1.0),
            goal: tuple[float, float] = (9.0, 9.0),
        ) -> np.ndarray:
            t = np.linspace(0.0, 1.0, steps)
            start_arr = np.array(start, dtype=float)
            goal_arr = np.array(goal, dtype=float)
            line = start_arr + (goal_arr - start_arr) * t[:, None]
            wiggle = np.column_stack(
                (
                    0.55 * np.sin(np.pi * t),
                    0.45 * np.sin(2.0 * np.pi * t + 0.4),
                )
            )
            path = line + wiggle
            return np.clip(path, 0.5, 9.5)

        def make_grid(grid_size: int, extent: tuple[float, float, float, float] = (0, 10, 0, 10)):
            xmin, xmax, ymin, ymax = extent
            xs = np.linspace(xmin, xmax, grid_size)
            ys = np.linspace(ymin, ymax, grid_size)
            X, Y = np.meshgrid(xs, ys)
            points = np.column_stack((X.ravel(), Y.ravel()))
            return xs, ys, points

        def gaussian_kernel(points: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
            sigma = max(float(sigma), 1e-6)
            delta = points - mean
            sq = np.sum(delta * delta, axis=1)
            return np.exp(-0.5 * sq / (sigma * sigma))

        def init_belief(points: np.ndarray, mean: np.ndarray, sigma: float) -> np.ndarray:
            belief = gaussian_kernel(points, mean, sigma)
            return belief / belief.sum()

        def predict_histogram(prior: np.ndarray, points: np.ndarray, control: np.ndarray, sigma: float) -> np.ndarray:
            sigma = max(float(sigma), 1e-6)
            shifted = points[:, None, :] + control[None, None, :]
            delta = points[None, :, :] - shifted
            sq = np.sum(delta * delta, axis=2)
            kernel = np.exp(-0.5 * sq / (sigma * sigma))
            predicted = prior @ kernel
            return predicted / predicted.sum()

        def measurement_likelihood(
            points: np.ndarray,
            measurement: np.ndarray,
            sigma: float,
            mode: str,
        ) -> np.ndarray:
            sigma = max(float(sigma), 1e-6)
            if mode == "x-only":
                sq = (points[:, 0] - float(measurement[0])) ** 2
            else:
                delta = points - measurement
                sq = np.sum(delta * delta, axis=1)
            return np.exp(-0.5 * sq / (sigma * sigma))

        def weighted_mean(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
            return weights @ points

        def entropy(weights: np.ndarray) -> float:
            safe = np.clip(weights, 1e-12, None)
            return float(-(safe * np.log(safe)).sum())

        def simulate_measurements(
            true_positions: np.ndarray,
            sigma: float,
            mode: str,
            rng: np.random.Generator,
        ) -> np.ndarray:
            sigma = max(float(sigma), 1e-6)
            if mode == "x-only":
                return true_positions[:, [0]] + rng.normal(0.0, sigma, size=(len(true_positions), 1))
            return true_positions + rng.normal(0.0, sigma, size=true_positions.shape)

        def run_demo(
            *,
            steps: int = 25,
            grid_size: int = 25,
            motion_sigma: float = 0.55,
            measurement_sigma: float = 0.7,
            prior_sigma: float = 0.95,
            measurement_mode: str = "x-only",
            seed: int = 7,
        ) -> DemoResults:
            rng = np.random.default_rng(seed)
            xs, ys, points = make_grid(grid_size)
            true_positions = build_trajectory(steps)
            controls = np.zeros_like(true_positions)
            controls[1:] = true_positions[1:] - true_positions[:-1]
            measurements = simulate_measurements(true_positions, measurement_sigma, measurement_mode, rng)

            start_mean = true_positions[0]
            markov_belief = init_belief(points, start_mean, prior_sigma)
            bayes_belief = init_belief(points, start_mean, prior_sigma)

            k_mean = start_mean.copy()
            k_cov = np.diag([prior_sigma**2, prior_sigma**2]).astype(float)
            q = np.diag([motion_sigma**2, motion_sigma**2]).astype(float)
            if measurement_mode == "x-only":
                H = np.array([[1.0, 0.0]])
                R = np.array([[measurement_sigma**2]])
            else:
                H = np.eye(2)
                R = np.diag([measurement_sigma**2, measurement_sigma**2])

            markov_beliefs = np.zeros((steps, grid_size, grid_size))
            bayes_beliefs = np.zeros((steps, grid_size, grid_size))
            markov_means = np.zeros((steps, 2))
            bayes_means = np.zeros((steps, 2))
            kalman_means = np.zeros((steps, 2))
            kalman_covs = np.zeros((steps, 2, 2))
            markov_errors = np.zeros(steps)
            bayes_errors = np.zeros(steps)
            kalman_errors = np.zeros(steps)
            markov_entropy = np.zeros(steps)
            bayes_entropy = np.zeros(steps)
            kalman_trace = np.zeros(steps)

            for t in range(steps):
                if t > 0:
                    markov_belief = predict_histogram(markov_belief, points, controls[t], motion_sigma)
                    bayes_belief = predict_histogram(bayes_belief, points, controls[t], motion_sigma)
                    k_mean = k_mean + controls[t]
                    k_cov = k_cov + q

                likelihood = measurement_likelihood(points, measurements[t], measurement_sigma, measurement_mode)
                bayes_belief = bayes_belief * likelihood
                bayes_belief = bayes_belief / bayes_belief.sum()

                z = measurements[t]
                innovation = z - H @ k_mean
                S = H @ k_cov @ H.T + R
                K = k_cov @ H.T @ np.linalg.inv(S)
                k_mean = k_mean + K @ innovation
                k_cov = (np.eye(2) - K @ H) @ k_cov

                markov_beliefs[t] = markov_belief.reshape(grid_size, grid_size)
                bayes_beliefs[t] = bayes_belief.reshape(grid_size, grid_size)
                markov_means[t] = weighted_mean(points, markov_belief)
                bayes_means[t] = weighted_mean(points, bayes_belief)
                kalman_means[t] = k_mean
                kalman_covs[t] = k_cov
                markov_errors[t] = np.linalg.norm(markov_means[t] - true_positions[t])
                bayes_errors[t] = np.linalg.norm(bayes_means[t] - true_positions[t])
                kalman_errors[t] = np.linalg.norm(kalman_means[t] - true_positions[t])
                markov_entropy[t] = entropy(markov_belief)
                bayes_entropy[t] = entropy(bayes_belief)
                kalman_trace[t] = float(np.trace(k_cov))

            return DemoResults(
                xs=xs,
                ys=ys,
                true_positions=true_positions,
                measurements=measurements,
                measurement_mode=measurement_mode,
                markov_beliefs=markov_beliefs,
                bayes_beliefs=bayes_beliefs,
                markov_means=markov_means,
                bayes_means=bayes_means,
                kalman_means=kalman_means,
                kalman_covs=kalman_covs,
                markov_errors=markov_errors,
                bayes_errors=bayes_errors,
                kalman_errors=kalman_errors,
                markov_entropy=markov_entropy,
                bayes_entropy=bayes_entropy,
                kalman_trace=kalman_trace,
            )

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
        options=["21", "25", "31"],
        value="25",
        label="Grid resolution",
    )
    motion_sigma = mo.ui.slider(0.15, 1.2, value=0.55, step=0.05, label="Motion noise")
    measurement_sigma = mo.ui.slider(0.15, 1.5, value=0.7, step=0.05, label="Measurement noise")
    prior_sigma = mo.ui.slider(0.25, 1.8, value=0.95, step=0.05, label="Initial uncertainty")
    measurement_mode = mo.ui.dropdown(
        options=["x-only sensor", "full 2D sensor"],
        value="x-only sensor",
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
    measurement_mode_value = "x-only" if measurement_mode.value == "x-only sensor" else "xy"
    results = run_demo(
        steps=steps.value,
        grid_size=int(grid_size.value),
        motion_sigma=motion_sigma.value,
        measurement_sigma=measurement_sigma.value,
        prior_sigma=prior_sigma.value,
        measurement_mode=measurement_mode_value,
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
    _fig, _axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
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

    for ax, title, belief, mean in zip(_axes[:2], titles[:2], beliefs, means[:2]):
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

    ax = _axes[2]
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

    _axes[0].legend(loc="upper right", fontsize=8)
    return _fig


@app.cell
def _(plt, results):
    _fig, _axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
    t = range(len(results.true_positions))

    _axes[0].plot(t, results.markov_errors, label="Markov only", linewidth=2)
    _axes[0].plot(t, results.bayes_errors, label="Bayes filter", linewidth=2)
    _axes[0].plot(t, results.kalman_errors, label="Kalman filter", linewidth=2)
    _axes[0].set_title("Position error over time")
    _axes[0].set_xlabel("time step")
    _axes[0].set_ylabel("Euclidean error")
    _axes[0].grid(alpha=0.25)
    _axes[0].legend()

    _axes[1].plot(t, results.markov_entropy, label="Markov entropy", linewidth=2)
    _axes[1].plot(t, results.bayes_entropy, label="Bayes entropy", linewidth=2)
    _axes[1].plot(t, results.kalman_trace, label="KF covariance trace", linewidth=2)
    _axes[1].set_title("Uncertainty trend")
    _axes[1].set_xlabel("time step")
    _axes[1].set_ylabel("uncertainty metric")
    _axes[1].grid(alpha=0.25)
    _axes[1].legend()
    return _fig


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
