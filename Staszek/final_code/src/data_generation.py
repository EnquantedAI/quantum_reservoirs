# src/data_generation.py

import numpy as np


def mackey_glass(beta=0.2, gamma=0.1, n=10, tau=30, dt=1.0, T=2000):
    """
    Generates the raw Mackey-Glass time series.

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
        The generated Mackey-Glass time series.
    """
    N = int(T / dt)
    delay_steps = int(tau / dt)
    x = np.zeros(N + delay_steps)
    x[0:delay_steps] = 1.2

    for t in range(delay_steps - 1, N + delay_steps - 1):
        x_tau = x[t - delay_steps]
        dxdt = (beta * x_tau / (1 + x_tau**n)) - (gamma * x[t])
        x[t+1] = x[t] + dxdt * dt

    return x[delay_steps:]


def split_time_series(time_series, train_fraction):
    """
    Splits a time series into train and test segments.

    Parameters
    ----------
    time_series : np.ndarray
        The full input time series.
    train_fraction : float
        Fraction of the series to allocate to training.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The train and test segments.
    """
    split_point = int(len(time_series) * train_fraction)
    time_series = np.asarray(time_series, dtype=float)
    return time_series[:split_point], time_series[split_point:]


def fit_minmax_scaler(train_series):
    """
    Fits min-max scaling statistics on the training split only.

    Parameters
    ----------
    train_series : np.ndarray
        The training time series used to estimate the scaling parameters.

    Returns
    -------
    dict[str, float]
        A mapping containing the training minimum and maximum.
    """
    train_series = np.asarray(train_series, dtype=float)
    return {
        "min": float(np.min(train_series)),
        "max": float(np.max(train_series)),
    }


def transform_with_minmax_scaler(time_series, scaler):
    """
    Applies pre-fit min-max scaling to a time series.

    Parameters
    ----------
    time_series : np.ndarray
        The series to transform.
    scaler : dict[str, float]
        The scaling statistics returned by ``fit_minmax_scaler``.

    Returns
    -------
    np.ndarray
        The transformed series.
    """
    time_series = np.asarray(time_series, dtype=float)
    scale_range = scaler["max"] - scaler["min"]
    if scale_range <= 0:
        return np.zeros_like(time_series, dtype=float)
    return (time_series - scaler["min"]) / scale_range


def split_and_scale_series(time_series, train_fraction):
    """
    Splits a time series and applies train-fit min-max scaling to both splits.

    Parameters
    ----------
    time_series : np.ndarray
        The full input time series.
    train_fraction : float
        Fraction of the series to allocate to training.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, float]]
        The scaled train split, scaled test split, and scaling statistics.
    """
    train_series, test_series = split_time_series(time_series, train_fraction)
    scaler = fit_minmax_scaler(train_series)
    return (
        transform_with_minmax_scaler(train_series, scaler),
        transform_with_minmax_scaler(test_series, scaler),
        scaler,
    )


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
    Generates raw time series data from an ARMA(p,q) process.

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
        The generated ARMA time series.
    """
    from scipy.signal import lfilter

    np.random.seed(seed)
    noise = np.random.normal(0, 1, n_points)
    data = lfilter(ma_coeffs, ar_coeffs, noise)

    return data


def generate_narma_data(n_points=2000, order=10, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1, seed=42):
    """
    Generates raw time series data from a NARMA process.

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
        The generated NARMA time series.
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

    return y
