import numpy as np


def mackey_glass(beta=0.2, gamma=0.1, n=10, tau=30, dt=1.0, T=2000):
    """Generate a Mackey-Glass time series."""
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
    """Split a series into train and test parts."""
    split_point = int(len(time_series) * train_fraction)
    time_series = np.asarray(time_series, dtype=float)
    return time_series[:split_point], time_series[split_point:]


def fit_minmax_scaler(train_series):
    """Fit min-max statistics on the training split."""
    train_series = np.asarray(train_series, dtype=float)
    return {
        "min": float(np.min(train_series)),
        "max": float(np.max(train_series)),
    }


def transform_with_minmax_scaler(time_series, scaler):
    """Apply a fitted min-max scaler to a series."""
    time_series = np.asarray(time_series, dtype=float)
    scale_range = scaler["max"] - scaler["min"]
    if scale_range <= 0:
        return np.zeros_like(time_series, dtype=float)
    return (time_series - scaler["min"]) / scale_range


def split_and_scale_series(time_series, train_fraction):
    """Split a series and scale both parts from the train split."""
    train_series, test_series = split_time_series(time_series, train_fraction)
    scaler = fit_minmax_scaler(train_series)
    return (
        transform_with_minmax_scaler(train_series, scaler),
        transform_with_minmax_scaler(test_series, scaler),
        scaler,
    )


def create_io_pairs(data, window_size, lag=0):
    """Build input-output pairs from a time series."""
    inputs, outputs = [], []
    for i in range(len(data) - window_size - lag):
        input_window = data[i : i + window_size]
        output_point = data[i + window_size + lag]
        inputs.append(input_window)
        outputs.append(output_point)

    return np.array(inputs), np.array(outputs)


def generate_arma_data(n_points=1000, ar_coeffs=[1, -0.7], ma_coeffs=[1, 0.5, -0.3], seed=42):
    """Generate an ARMA time series."""
    from scipy.signal import lfilter

    np.random.seed(seed)
    noise = np.random.normal(0, 1, n_points)
    data = lfilter(ma_coeffs, ar_coeffs, noise)

    return data


def generate_narma_data(n_points=2000, order=10, alpha=0.3, beta=0.05, gamma=1.5, delta=0.1, seed=42):
    """Generate a NARMA time series."""
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
