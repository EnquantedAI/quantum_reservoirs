# src/data_generation.py

import numpy as np
from scipy.signal import lfilter


def mackey_glass(beta=0.2, gamma=0.1, n=10, tau=30, dt=1.0, T=2000):
    """
    Generates the Mackey-Glass time series and scales it to [0, 1].

    Parameters
    ----------
    beta : float, optional
        Equation parameter, by default 0.2.
    gamma : float, optional
        Equation parameter, by default 0.1.
    n : int, optional
        Equation parameter, by default 10.
    tau : int, optional
        Time delay parameter, by default 30.
    dt : float, optional
        Time step size, by default 1.0.
    T : int, optional
        Total time length, by default 2000.

    Returns
    -------
    np.ndarray
        The generated and scaled Mackey-Glass time series.
    """
    N = int(T / dt)
    delay_steps = int(tau / dt)
    x = np.zeros(N + delay_steps)
    x[0:delay_steps] = 1.2

    for t in range(delay_steps - 1, N + delay_steps - 1):
        x_tau = x[t - delay_steps]
        dxdt = (beta * x_tau / (1 + x_tau**n)) - (gamma * x[t])
        x[t+1] = x[t] + dxdt * dt

    x_series = x[delay_steps:]
    x_min, x_max = np.min(x_series), np.max(x_series)
    x_scaled = (x_series - x_min) / (x_max - x_min)

    return x_scaled


def create_io_pairs(data, window_size, lag=0):
    """
    Creates input-output pairs from a time series for supervised learning.

    Parameters
    ----------
    data : np.ndarray
        The input time series.
    window_size : int
        The length of the input window (number of features).
    lag : int, optional
        The time lag between the end of the input window and the output, by default 0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two arrays: inputs (X) and outputs (y).
    """
    inputs, outputs = [], []
    for i in range(len(data) - window_size - lag):
        input_window = data[i : i + window_size]
        output_point = data[i + window_size + lag]
        inputs.append(input_window)
        outputs.append(output_point)

    return np.array(inputs), np.array(outputs)


def generate_arma_data(n_points=1000, ar_coeffs=[1, -0.7], ma_coeffs=[1, 0.5, -0.3], seed=42):
    """
    Generates time series data from an ARMA(p,q) process and scales it to [0, 1].

    Parameters
    ----------
    n_points : int, optional
        Number of data points to generate, by default 1000.
    ar_coeffs : list, optional
        Coefficients of the autoregressive (AR) part, by default [1, -0.7].
    ma_coeffs : list, optional
        Coefficients of the moving average (MA) part, by default [1, 0.5, -0.3].
    seed : int, optional
        Seed for the random number generator, by default 42.

    Returns
    -------
    np.ndarray
        The generated and scaled ARMA time series.
    """
    np.random.seed(seed)
    noise = np.random.normal(0, 1, n_points)
    data = lfilter(ma_coeffs, ar_coeffs, noise)

    data_min, data_max = np.min(data), np.max(data)
    data_scaled = (data - data_min) / (data_max - data_min)

    return data_scaled


def generate_narma_data(n_points=2000, order=10, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1, seed=42):
    """
    Generates time series data from a NARMA process and scales it to [0, 1].

    Parameters
    ----------
    n_points : int, optional
        Total number of points in the time series, by default 2000.
    order : int, optional
        The memory order of the system (e.g., 10 for NARMA10), by default 10.
    alpha, beta, gamma, delta : float, optional
        Coefficients of the NARMA equation.
    seed : int, optional
        Seed for the random number generator, by default 42.

    Returns
    -------
    np.ndarray
        The generated and scaled NARMA time series.
    """
    np.random.seed(seed)
    s = np.random.uniform(0, 0.5, n_points)
    y = np.zeros(n_points)

    for k in range(order, n_points):
        sum_term = np.sum(y[k-order:k])
        y[k] = (alpha * y[k-1] +
                beta * y[k-1] * sum_term +
                gamma * s[k-order] * s[k] +
                delta)

    y_min, y_max = np.min(y), np.max(y)
    y_scaled = (y - y_min) / (y_max - y_min)

    return y_scaled