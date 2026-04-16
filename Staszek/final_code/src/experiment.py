# src/experiment.py

import numpy as np
from sklearn.metrics import mean_squared_error

from .data_generation import create_io_pairs, split_and_scale_series
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)

# --- Podstawowe funkcje uruchamiające pojedynczy eksperyment (bez zmian) ---

DEFAULT_EVAL_PROTOCOL = "cold_start"
DEFAULT_CLASSICAL_LAG = 0


def _parse_classical_params(params):
    """Supports both the new and legacy classical parameter tuples."""
    if len(params) == 6:
        reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg, window_size = params
    elif len(params) == 5:
        reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg = params
        window_size = 10
    else:
        raise ValueError(f"Unexpected classical parameter set: {params}")

    return (
        reservoir_size,
        spectral_radius,
        sparsity,
        leakage_rate,
        lambda_reg,
        int(window_size),
        DEFAULT_CLASSICAL_LAG,
    )


def _select_representative_seed(sub_seeds, mse_scores):
    """Selects the seed corresponding to the median-ranked trial."""
    ranked_indices = np.argsort(np.asarray(mse_scores), kind="stable")
    representative_index = ranked_indices[len(ranked_indices) // 2]
    return sub_seeds[representative_index]

def run_single_qrc_trial(params, profile, time_series, train_fraction, seed):
    """Runs a SINGLE trial for the QRC model for one specific seed."""
    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    train_data, test_data, _ = split_and_scale_series(time_series, train_fraction)
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
    (reservoir_size, spectral_radius, sparsity, leakage_rate,
     lambda_reg, window_size, lag) = _parse_classical_params(params)
    train_data, test_data, _ = split_and_scale_series(time_series, train_fraction)
    train_inputs, train_outputs = create_io_pairs(train_data, window_size, lag)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size, lag)
    
    W_in, W_res = initialize_classical_reservoir(reservoir_size, window_size, spectral_radius, sparsity, seed)
    W_out, _diagnostic_final_state = train_classical_reservoir(
        train_inputs, train_outputs, W_in, W_res,
        reservoir_size, leakage_rate, lambda_reg
    )
    predictions = predict_esn_classical(
        test_inputs, W_in, W_res, W_out,
        reservoir_size, leakage_rate
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
    representative_seed = _select_representative_seed(sub_seeds, mse_scores)

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
        'base_seed': base_seed,
        'representative_seed': representative_seed,
        'num_trials': num_trials,
        'eval_protocol': DEFAULT_EVAL_PROTOCOL,
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
    representative_seed = _select_representative_seed(sub_seeds, mse_scores)

    (reservoir_size, spectral_radius, sparsity, leakage_rate,
     lambda_reg, window_size, lag) = _parse_classical_params(params)
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
        'window_size': window_size,
        'lag': lag,
        'base_seed': base_seed,
        'representative_seed': representative_seed,
        'num_trials': num_trials,
        'eval_protocol': DEFAULT_EVAL_PROTOCOL,
    }
