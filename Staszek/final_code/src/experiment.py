# src/experiment.py

import numpy as np
from sklearn.metrics import mean_squared_error

from .data_generation import create_io_pairs
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)

# --- Podstawowe funkcje uruchamiające pojedynczy eksperyment (bez zmian) ---

def run_single_qrc_trial(params, profile, time_series, train_fraction, seed):
    """Runs a SINGLE trial for the QRC model for one specific seed."""
    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    split_point = int(len(time_series) * train_fraction)
    train_data, test_data = time_series[:split_point], time_series[split_point:]
    train_inputs, train_outputs = create_io_pairs(train_data, window_size, lag)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size, lag)
    
    W_out, weights, biases, _ = train_esn_reservoir(
        train_inputs, train_outputs, n_layers, window_size,
        leakage_rate, lambda_reg, seed
    )
    predictions = predict_esn(
        test_inputs, weights, biases, W_out, n_layers,
        window_size, leakage_rate
    )
    return mean_squared_error(test_outputs, predictions)

def run_single_classical_trial(params, profile, time_series, train_fraction, seed):
    """Runs a SINGLE trial for the Classical ESN model for one specific seed."""
    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg = params
    window_size = 10
    split_point = int(len(time_series) * train_fraction)
    train_data, test_data = time_series[:split_point], time_series[split_point:]
    train_inputs, train_outputs = create_io_pairs(train_data, window_size)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size)
    
    W_in, W_res = initialize_classical_reservoir(reservoir_size, window_size, spectral_radius, sparsity, seed)
    W_out, final_state = train_classical_reservoir(
        train_inputs, train_outputs, W_in, W_res,
        reservoir_size, leakage_rate, lambda_reg
    )
    predictions = predict_esn_classical(
        test_inputs, W_in, W_res, W_out,
        reservoir_size, leakage_rate, final_state
    )
    return mean_squared_error(test_outputs, predictions)

# --- NOWE FUNKCJE-WRAPPERS, KTÓRE ZARZĄDZAJĄ POD-ZIARNAMI ---

def run_qrc_experiment_with_subseeds(params, profile, time_series, train_fraction, base_seed, num_trials=11):
    """
    Runs a QRC experiment multiple times with different sub-seeds and returns aggregated results.
    """
    mse_scores = []
    # Tworzymy listę pod-ziaren na podstawie głównego ziarna
    sub_seeds = [base_seed + i for i in range(num_trials)]
    
    for seed in sub_seeds:
        mse = run_single_qrc_trial(params, profile, time_series, train_fraction, seed)
        mse_scores.append(mse)
    
    # Obliczamy statystyki
    median_mse = np.median(mse_scores)
    std_dev_mse = np.std(mse_scores)
    # Współczynnik zmienności (CV) - miara stabilności
    cv_mse = std_dev_mse / median_mse if median_mse > 0 else 0

    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    return {
        'model_type': 'QRC',
        'data_profile': profile['name'],
        'median_mse': median_mse,
        'std_dev_mse': std_dev_mse,
        'cv_mse': cv_mse,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': window_size,
        'n_layers': n_layers,
        'lag': lag,
        'base_seed': base_seed
    }

def run_classical_experiment_with_subseeds(params, profile, time_series, train_fraction, base_seed, num_trials=11):
    """
    Runs a Classical ESN experiment multiple times and returns aggregated results.
    """
    mse_scores = []
    sub_seeds = [base_seed + i for i in range(num_trials)]

    for seed in sub_seeds:
        mse = run_single_classical_trial(params, profile, time_series, train_fraction, seed)
        mse_scores.append(mse)

    median_mse = np.median(mse_scores)
    std_dev_mse = np.std(mse_scores)
    cv_mse = std_dev_mse / median_mse if median_mse > 0 else 0

    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg = params
    return {
        'model_type': 'Classical_ESN',
        'data_profile': profile['name'],
        'median_mse': median_mse,
        'std_dev_mse': std_dev_mse,
        'cv_mse': cv_mse,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'sparsity': sparsity,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': 10,
        'base_seed': base_seed
    }