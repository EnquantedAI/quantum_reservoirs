# src/analysis.py

import numpy as np
from sklearn.decomposition import PCA
from .data_generation import create_io_pairs
from .models import train_esn_reservoir, get_classical_reservoir_states

def get_qrc_feature_space(params, time_series, train_fraction, seed):
    """
    Generates the quantum feature space for a given set of QRC parameters.
    """
    leakage, lambda_r, win_size, layers, lag = params
    n_qubits = win_size
    train_size = int(len(time_series) * train_fraction)
    
    train_inputs, train_outputs = create_io_pairs(time_series[:train_size], win_size, lag)
    
    # We only need the quantum_features, so we ignore the other return values
    _, _, _, quantum_features = train_esn_reservoir(
        train_inputs, train_outputs, layers, n_qubits, leakage, lambda_r, seed
    )
    return quantum_features

def calculate_effective_dimension(feature_matrix, variance_threshold=0.95):
    """
    Calculates the effective dimensionality of a feature space using PCA.
    
    The effective dimension is the number of principal components needed to
    explain a certain amount of the total variance.
    """
    if feature_matrix is None or feature_matrix.shape[0] < 2:
        return np.nan # Cannot perform PCA on empty or single-sample data
        
    pca = PCA()
    pca.fit(feature_matrix)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the first index where cumulative variance exceeds the threshold
    eff_dim = np.argmax(cumulative_variance >= variance_threshold) + 1
    return eff_dim