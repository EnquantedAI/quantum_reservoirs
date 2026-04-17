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
