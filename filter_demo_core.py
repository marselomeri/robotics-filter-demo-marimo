from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    total = belief.sum()
    return belief / total


def predict_histogram(prior: np.ndarray, points: np.ndarray, control: np.ndarray, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    shifted = points[:, None, :] + control[None, None, :]
    delta = points[None, :, :] - shifted
    sq = np.sum(delta * delta, axis=2)
    kernel = np.exp(-0.5 * sq / (sigma * sigma))
    predicted = prior @ kernel
    total = predicted.sum()
    return predicted / total


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
