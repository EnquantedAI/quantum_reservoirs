import warnings

import numpy as np
from .data_generation import create_io_pairs, split_and_scale_series
from .models import train_esn_reservoir, get_classical_reservoir_states

def get_qrc_feature_space(params, time_series, train_fraction, seed):
    """Return QRC features for the training split."""
    leakage, lambda_r, win_size, layers, lag = params
    n_qubits = win_size
    train_series, _, _ = split_and_scale_series(time_series, train_fraction)
    train_inputs, train_outputs = create_io_pairs(train_series, win_size, lag)
    
    # Keep only the feature matrix.
    _, _, _, quantum_features = train_esn_reservoir(
        train_inputs, train_outputs, layers, n_qubits, leakage, lambda_r, seed
    )
    return quantum_features

def calculate_participation_ratio(feature_matrix):
    """Compute the participation ratio of a feature matrix."""
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    if feature_matrix.size == 0 or feature_matrix.ndim != 2 or feature_matrix.shape[0] < 2:
        return np.nan
    if not np.all(np.isfinite(feature_matrix)):
        return np.nan

    if np.allclose(np.var(feature_matrix, axis=0), 0.0):
        return np.nan

    from sklearn.decomposition import PCA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pca = PCA()
        pca.fit(feature_matrix)

    eigenvalues = np.asarray(pca.explained_variance_, dtype=float)
    if eigenvalues.size == 0 or not np.all(np.isfinite(eigenvalues)):
        return np.nan

    eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
    eigenvalue_sum = np.sum(eigenvalues)
    eigenvalue_sq_sum = np.sum(eigenvalues ** 2)
    if eigenvalue_sum <= 0 or eigenvalue_sq_sum <= 0:
        return np.nan

    return (eigenvalue_sum ** 2) / eigenvalue_sq_sum


def calculate_effective_dimension(feature_matrix, variance_threshold=0.95):
    """Backward-compatible alias for participation ratio."""
    return calculate_participation_ratio(feature_matrix)


def _ridge_fit(X, y, lambda_reg):
    """Ridge solver that matches the readout pattern in models.py."""
    d = X.shape[1]
    I_mat = np.identity(d)
    return np.linalg.solve(X.T @ X + lambda_reg * I_mat, X.T @ y)


def memory_capacity_spectrum(features, reference_series, lags, lambda_reg=1e-8,
                             train_fraction=0.7):
    """Per-lag R^2 for recovering reference_series[t-k] from features[t].

    features and reference_series share the same time index; row t of features
    is aligned to reference_series[t]. For each lag k, a ridge probe is fit on
    the first train_fraction of the aligned samples and scored on the rest.
    Returns (lags_array, mc_array). Values are clipped to [0, 1] so the
    spectrum reads as recoverable variance.
    """
    features = np.asarray(features, dtype=float)
    reference_series = np.asarray(reference_series, dtype=float).ravel()
    if features.ndim != 2 or features.shape[0] != reference_series.shape[0]:
        raise ValueError("features and reference_series must share leading axis")

    lags = np.asarray(list(lags), dtype=int)
    n = features.shape[0]
    mc = np.full(lags.shape[0], np.nan, dtype=float)

    for i, k in enumerate(lags):
        if k < 0 or k >= n:
            continue
        X = features[k:]
        y = reference_series[: n - k]
        m = X.shape[0]
        split = int(train_fraction * m)
        if split < 2 or m - split < 1:
            continue
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        try:
            W = _ridge_fit(X_train, y_train, lambda_reg)
        except np.linalg.LinAlgError:
            continue

        y_pred = X_test @ W
        var_y = float(np.var(y_test))
        if not np.isfinite(var_y) or var_y <= 0:
            continue
        mse = float(np.mean((y_test - y_pred) ** 2))
        mc[i] = max(0.0, min(1.0, 1.0 - mse / var_y))

    return lags, mc


def canonical_correlations(F_a, F_b, n_components=None):
    """Top canonical correlation coefficients between two feature matrices.

    Uses QR/SVD whitening so it stays numerically stable when either matrix
    has more columns than rows. Returns an array of correlations in [0, 1]
    sorted in descending order.
    """
    F_a = np.asarray(F_a, dtype=float)
    F_b = np.asarray(F_b, dtype=float)
    if F_a.ndim != 2 or F_b.ndim != 2 or F_a.shape[0] != F_b.shape[0]:
        raise ValueError("F_a and F_b must be 2D and share leading axis")

    A = F_a - F_a.mean(axis=0, keepdims=True)
    B = F_b - F_b.mean(axis=0, keepdims=True)

    # Drop constant columns so QR does not produce a zero pivot.
    A = A[:, np.any(A != 0.0, axis=0)]
    B = B[:, np.any(B != 0.0, axis=0)]
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.array([])

    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    singular_values = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    corrs = np.clip(singular_values, 0.0, 1.0)

    if n_components is not None:
        corrs = corrs[:n_components]
    return corrs
