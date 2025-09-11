# src/visualization.py

import matplotlib.pyplot as plt
import numpy as np

# Import necessary functions from other modules
from .data_generation import create_io_pairs
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)


def plot_best_model_comparison(best_qrc_row, best_classical_row, data_profile_config, constants):
    """
    Generates and displays a plot comparing the best QRC and Classical ESN models for a given data profile.

    This function performs the following steps:
    1.  Regenerates the original time series for the profile.
    2.  Splits the data into training and testing sets.
    3.  Re-trains the best QRC model using its optimal hyperparameters.
    4.  Re-trains the best Classical ESN model using its optimal hyperparameters.
    5.  Generates one-step-ahead predictions from both models.
    6.  Plots the true test data against the predictions from both models.

    Parameters
    ----------
    best_qrc_row : pd.Series
        The row from the results DataFrame corresponding to the best QRC model.
    best_classical_row : pd.Series
        The row from the results DataFrame corresponding to the best Classical ESN model.
    data_profile_config : dict
        The dictionary defining the data profile (name, generator, params).
    constants : dict
        A dictionary containing global constants like TRAIN_FRACTION and SEED.
    """
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating plot for profile: {profile_name} ---\n")

    # --- 1. Regenerate Time Series ---
    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    train_size = int(len(time_series) * constants['TRAIN_FRACTION'])
    train_series, test_series = time_series[:train_size], time_series[train_size:]

    # --- 2. Re-train Best QRC Model ---
    print(f"Retraining best QRC model...")
    qrc_win_size = int(best_qrc_row['window_size'])
    qrc_train_inputs, qrc_train_outputs = create_io_pairs(train_series, qrc_win_size)
    qrc_test_inputs, qrc_test_outputs = create_io_pairs(test_series, qrc_win_size)

    W_out_q, weights_q, biases_q, _ = train_esn_reservoir(
        qrc_train_inputs, qrc_train_outputs,
        n_layers=int(best_qrc_row['n_layers']),
        n_qubits=qrc_win_size,
        leakage_rate=best_qrc_row['leakage_rate'],
        lambda_reg=best_qrc_row['lambda_reg'],
        seed=constants['SEED']
    )
    qrc_preds = predict_esn(
        qrc_test_inputs, weights_q, biases_q, W_out_q,
        int(best_qrc_row['n_layers']), qrc_win_size,
        best_qrc_row['leakage_rate']
    )
    print(f"Best QRC MSE: {best_qrc_row['mse']:.6f}")

    # --- 3. Re-train Best Classical ESN Model ---
    print(f"Retraining best Classical ESN model...")
    classical_win_size = int(best_classical_row['window_size'])
    classical_train_inputs, classical_train_outputs = create_io_pairs(train_series, classical_win_size)
    classical_test_inputs, classical_test_outputs = create_io_pairs(test_series, classical_win_size)

    W_in_c, W_res_c = initialize_classical_reservoir(
        reservoir_size=int(best_classical_row['reservoir_size']),
        input_dim=classical_win_size,
        spectral_radius=best_classical_row['spectral_radius'],
        sparsity=best_classical_row['sparsity'],
        seed=constants['SEED']
    )
    W_out_c, last_state_c = train_classical_reservoir(
        classical_train_inputs, classical_train_outputs, W_in_c, W_res_c,
        reservoir_size=int(best_classical_row['reservoir_size']),
        leakage_rate=best_classical_row['leakage_rate'],
        lambda_reg=best_classical_row['lambda_reg']
    )
    classical_preds = predict_esn_classical(
        classical_test_inputs, W_in_c, W_res_c, W_out_c,
        reservoir_size=int(best_classical_row['reservoir_size']),
        leakage_rate=best_classical_row['leakage_rate'],
        initial_state=last_state_c
    )
    print(f"Best Classical ESN MSE: {best_classical_row['mse']:.6f}")

    # --- 4. Plot Comparison ---
    # Ensure all arrays are of the same length for plotting
    min_len = min(len(qrc_test_outputs), len(classical_test_outputs))
    test_outputs = qrc_test_outputs[:min_len]

    plt.figure(figsize=(15, 7))
    plt.plot(test_outputs, label="True Data (Test Set)", color="blue", linewidth=2.5, alpha=0.8)
    plt.plot(qrc_preds[:min_len], label=f"Best QRC Prediction (MSE: {best_qrc_row['mse']:.6f})", color="#e41a1c", linestyle="--", alpha=0.9)
    plt.plot(classical_preds[:min_len], label=f"Best ESN Prediction (MSE: {best_classical_row['mse']:.6f})", color="#4daf4a", linestyle=":", alpha=0.9)

    plt.xlabel("Time Step (in test set)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.title(f"One-Step-Ahead Prediction Comparison for: {profile_name}", fontsize=14, weight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save the figure to the reports/figures directory
    plt.savefig(f"../reports/figures/{profile_name}_comparison.png", dpi=300)
    plt.show()