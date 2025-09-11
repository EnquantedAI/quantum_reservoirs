# src/experiment.py

import numpy as np
from sklearn.metrics import mean_squared_error

from .data_generation import create_io_pairs
from .models import (train_esn_reservoir, predict_esn, 
                     initialize_classical_reservoir, train_classical_reservoir, 
                     predict_esn_classical)

def run_qrc_experiment(params, profile, time_series, train_fraction, seed):
    """
    Runs a single experiment for the QRC-ESN model with a given set of parameters.
    """
    # Unpack parameters
    leakage_rate, lambda_reg, window_size, n_layers, lag = params
    
    # Split data
    split_point = int(len(time_series) * train_fraction)
    train_data, test_data = time_series[:split_point], time_series[split_point:]
    
    # Create input-output pairs
    train_inputs, train_outputs = create_io_pairs(train_data, window_size, lag)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size, lag)
    
    # Train model
    W_out, weights, biases, _ = train_esn_reservoir(
        train_inputs, train_outputs, n_layers, window_size, 
        leakage_rate, lambda_reg, seed
    )
    
    # Make predictions
    predictions = predict_esn(
        test_inputs, weights, biases, W_out, n_layers, 
        window_size, leakage_rate
    )
    
    # Calculate error
    mse = mean_squared_error(test_outputs, predictions)
    
    # Return results as a dictionary
    return {
        'model_type': 'QRC',
        'data_profile': profile['name'],
        'mse': mse,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': window_size,
        'n_layers': n_layers,
        'lag': lag,
        'seed': seed
    }

def run_classical_experiment(params, profile, time_series, train_fraction, seed):
    """
    Runs a single experiment for the Classical ESN model.
    """
    # Unpack parameters
    reservoir_size, spectral_radius, sparsity, leakage_rate, lambda_reg = params
    window_size = 10 # Assuming a fixed window_size for classical ESN for simplicity
                     # This might need to be added to its param_grid if it needs to be varied

    # Split data
    split_point = int(len(time_series) * train_fraction)
    train_data, test_data = time_series[:split_point], time_series[split_point:]
    
    # Create I/O pairs
    train_inputs, train_outputs = create_io_pairs(train_data, window_size)
    test_inputs, test_outputs = create_io_pairs(test_data, window_size)
    
    # Initialize and train
    W_in, W_res = initialize_classical_reservoir(reservoir_size, window_size, spectral_radius, sparsity, seed)
    W_out, final_state = train_classical_reservoir(
        train_inputs, train_outputs, W_in, W_res, 
        reservoir_size, leakage_rate, lambda_reg
    )
    
    # Predict
    predictions = predict_esn_classical(
        test_inputs, W_in, W_res, W_out, 
        reservoir_size, leakage_rate, final_state
    )
    
    # Calculate error
    mse = mean_squared_error(test_outputs, predictions)
    
    return {
        'model_type': 'Classical_ESN',
        'data_profile': profile['name'],
        'mse': mse,
        'reservoir_size': reservoir_size,
        'spectral_radius': spectral_radius,
        'sparsity': sparsity,
        'leakage_rate': leakage_rate,
        'lambda_reg': lambda_reg,
        'window_size': window_size,
        'seed': seed
    }