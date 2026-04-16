# src/visualization.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Import necessary functions from other modules
from .data_generation import create_io_pairs, split_and_scale_series
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)


def _get_optional_int(row, key, default):
    """Converts a scalar result field into an int with NaN fallback support."""
    value = row.get(key, default)
    try:
        if np.isnan(value):
            return int(default)
    except TypeError:
        pass
    return int(value)


def _resolve_representative_seed(row, constants):
    """Uses the stored representative seed when available."""
    for key in ("representative_seed", "base_seed"):
        value = row.get(key)
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass
        if value is not None:
            return int(value)
    return int(constants.get("SEED", 2025))


def plot_best_model_comparison(best_qrc_row, best_classical_row, data_profile_config, constants, show=True):
    """
    Generates and displays a plot comparing the best QRC and Classical ESN models.
    """
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating plot for profile: {profile_name} ---\n")

    # --- 1. Regenerate Time Series ---
    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    train_series, test_series, _ = split_and_scale_series(time_series, constants['TRAIN_FRACTION'])
    figure_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # --- 2. Re-train Best QRC Model ---
    print(f"Retraining best QRC model...")
    qrc_win_size = int(best_qrc_row['window_size'])
    qrc_lag = _get_optional_int(best_qrc_row, 'lag', 0)
    qrc_seed = _resolve_representative_seed(best_qrc_row, constants)
    qrc_train_inputs, qrc_train_outputs = create_io_pairs(train_series, qrc_win_size, qrc_lag)
    qrc_test_inputs, qrc_test_outputs = create_io_pairs(test_series, qrc_win_size, qrc_lag)

    W_out_q, weights_q, biases_q, _ = train_esn_reservoir(
        qrc_train_inputs, qrc_train_outputs,
        n_layers=int(best_qrc_row['n_layers']),
        n_qubits=qrc_win_size,
        leakage_rate=best_qrc_row['leakage_rate'],
        lambda_reg=best_qrc_row['lambda_reg'],
        seed=qrc_seed
    )
    qrc_preds = predict_esn(
        qrc_test_inputs, weights_q, biases_q, W_out_q,
        int(best_qrc_row['n_layers']), qrc_win_size,
        best_qrc_row['leakage_rate']
    )
    qrc_mse = mean_squared_error(qrc_test_outputs, qrc_preds)
    print(
        f"Best QRC Median MSE: {best_qrc_row['median_mse']:.6f} | "
        f"Representative seed: {qrc_seed} | Reproduced MSE: {qrc_mse:.6f}"
    )

    # --- 3. Re-train Best Classical ESN Model ---
    # (We keep the training logic to allow console comparison, but it won't be plotted)
    print(f"Retraining best Classical ESN model...")
    classical_win_size = _get_optional_int(best_classical_row, 'window_size', 10)
    classical_lag = _get_optional_int(best_classical_row, 'lag', 0)
    classical_seed = _resolve_representative_seed(best_classical_row, constants)
    eval_protocol = str(best_classical_row.get('eval_protocol', 'cold_start'))
    classical_train_inputs, classical_train_outputs = create_io_pairs(train_series, classical_win_size, classical_lag)
    classical_test_inputs, classical_test_outputs = create_io_pairs(test_series, classical_win_size, classical_lag)

    W_in_c, W_res_c = initialize_classical_reservoir(
        reservoir_size=_get_optional_int(best_classical_row, 'reservoir_size', 10),
        input_dim=classical_win_size,
        spectral_radius=best_classical_row['spectral_radius'],
        sparsity=best_classical_row['sparsity'],
        seed=classical_seed
    )
    W_out_c, last_state_c = train_classical_reservoir(
        classical_train_inputs, classical_train_outputs, W_in_c, W_res_c,
        reservoir_size=_get_optional_int(best_classical_row, 'reservoir_size', 10),
        leakage_rate=best_classical_row['leakage_rate'],
        lambda_reg=best_classical_row['lambda_reg']
    )
    initial_state = None if eval_protocol == 'cold_start' else last_state_c
    classical_preds = predict_esn_classical(
        classical_test_inputs, W_in_c, W_res_c, W_out_c,
        reservoir_size=_get_optional_int(best_classical_row, 'reservoir_size', 10),
        leakage_rate=best_classical_row['leakage_rate'],
        initial_state=initial_state
    )
    classical_mse = mean_squared_error(classical_test_outputs, classical_preds)
    print(
        f"Best Classical ESN Median MSE: {best_classical_row['median_mse']:.6f} | "
        f"Representative seed: {classical_seed} | Reproduced MSE: {classical_mse:.6f}"
    )

    # --- 4. Plot Comparison ---
    # Limit to first 200 time steps
    plot_limit = 200
    min_len = min(len(qrc_test_outputs), plot_limit)
    
    test_outputs = qrc_test_outputs[:min_len]
    qrc_preds_plot = qrc_preds[:min_len]
    # Classical predictions prepared but NOT plotted

    plt.figure(figsize=(15, 7))
    
    # 1. True Data (Solid line)
    plt.plot(test_outputs, label="True Data (Test Set)", color="black", linewidth=2.5, alpha=0.8)
    
    # 2. QRC Prediction (Dashed line, BOLDED via linewidth=2.5)
    plt.plot(qrc_preds_plot, 
             label=f"Best QRC Prediction (Representative MSE: {qrc_mse:.6f})", 
             color="black", 
             linestyle="--", 
             alpha=0.9,
             linewidth=2.5) # Increased thickness
    
    # MODIFICATION: Classical ESN plot removed as requested.

    plt.xlabel("Time Step (in test set)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    plt.title(f"One-Step-Ahead Prediction Comparison for: {profile_name}", fontsize=14, weight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{profile_name}_comparison.png", dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
