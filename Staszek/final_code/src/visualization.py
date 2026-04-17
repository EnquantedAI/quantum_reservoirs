from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Import project helpers.
from .data_generation import create_io_pairs, split_and_scale_series
from .models import (train_esn_reservoir, predict_esn,
                     initialize_classical_reservoir, train_classical_reservoir,
                     predict_esn_classical)

VALID_PLOT_STYLES = {"color", "bw"}


def _get_optional_int(row, key, default):
    """Read an integer field with NaN fallback."""
    value = row.get(key, default)
    try:
        if np.isnan(value):
            return int(default)
    except TypeError:
        pass
    return int(value)


def _resolve_representative_seed(row, constants):
    """Resolve the stored seed, if present."""
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


def validate_plot_style(plot_style):
    """Validate and normalize the plot style flag."""
    normalized = str(plot_style).strip().lower()
    if normalized not in VALID_PLOT_STYLES:
        raise ValueError(
            f"Invalid plot style '{plot_style}'. Expected one of: {sorted(VALID_PLOT_STYLES)}."
        )
    return normalized


def get_plot_style_config(plot_style):
    """Return a consistent style mapping for notebook figures."""
    plot_style = validate_plot_style(plot_style)
    window_sizes = [1, 2, 4, 6, 8, 10]

    if plot_style == "color":
        window_palette = {
            1: "#4c78a8",
            2: "#f58518",
            4: "#54a24b",
            6: "#e45756",
            8: "#72b7b2",
            10: "#b279a2",
        }
        return {
            "plot_style": plot_style,
            "window_sizes": window_sizes,
            "window_palette": window_palette,
            "grid_alpha": 0.30,
            "annotation_facecolor": "#f8f9fb",
            "annotation_edgecolor": "#c7ccd6",
            "line_styles": {
                "truth": {"color": "#202124", "linestyle": "-", "linewidth": 2.5},
                "qrc": {"color": "#4c78a8", "linestyle": "--", "linewidth": 2.1},
                "classical": {"color": "#f58518", "linestyle": ":", "linewidth": 2.3},
                "qrc_error": {"color": "#4c78a8", "linestyle": "--", "linewidth": 1.8},
                "classical_error": {"color": "#f58518", "linestyle": "-", "linewidth": 1.8},
                "trend": {"color": "#7f7f7f", "linestyle": "--", "linewidth": 1.5},
            },
            "summary_colors": {"qrc": "#4c78a8", "classical": "#f58518"},
            "summary_hatches": {"qrc": "", "classical": ""},
            "classical_marker": {
                "marker": "^",
                "s": 180,
                "facecolor": "#f58518",
                "edgecolor": "#202124",
                "linewidth": 0.9,
            },
            "best_qrc_marker": {
                "marker": "o",
                "s": 130,
                "facecolor": "#4c78a8",
                "edgecolor": "#202124",
                "linewidth": 0.9,
            },
        }

    window_palette = {
        1: "#d9d9d9",
        2: "#bdbdbd",
        4: "#969696",
        6: "#737373",
        8: "#525252",
        10: "#252525",
    }
    return {
        "plot_style": plot_style,
        "window_sizes": window_sizes,
        "window_palette": window_palette,
        "grid_alpha": 0.42,
        "annotation_facecolor": "#fbfbfb",
        "annotation_edgecolor": "#6c6c6c",
        "line_styles": {
            "truth": {"color": "#111111", "linestyle": "-", "linewidth": 2.6},
            "qrc": {"color": "#3d3d3d", "linestyle": "--", "linewidth": 2.1},
            "classical": {"color": "#7d7d7d", "linestyle": ":", "linewidth": 2.5},
            "qrc_error": {"color": "#3d3d3d", "linestyle": "--", "linewidth": 1.8},
            "classical_error": {"color": "#6a6a6a", "linestyle": "-", "linewidth": 1.8},
            "trend": {"color": "#5f5f5f", "linestyle": "--", "linewidth": 1.5},
        },
        "summary_colors": {"qrc": "#2f2f2f", "classical": "#9a9a9a"},
        "summary_hatches": {"qrc": "///", "classical": ""},
        "classical_marker": {
            "marker": "^",
            "s": 180,
            "facecolor": "#111111",
            "edgecolor": "#111111",
            "linewidth": 0.9,
        },
        "best_qrc_marker": {
            "marker": "o",
            "s": 130,
            "facecolor": "#4f4f4f",
            "edgecolor": "#111111",
            "linewidth": 0.9,
        },
    }


def _align_predictions_to_common_positions(test_series, qrc_win_size, qrc_lag, qrc_targets,
                                           qrc_preds, classical_win_size, classical_lag,
                                           classical_targets, classical_preds, plot_limit):
    """Align model outputs on the shared absolute test-set positions."""
    qrc_positions = np.arange(qrc_win_size + qrc_lag, qrc_win_size + qrc_lag + len(qrc_targets))
    classical_positions = np.arange(
        classical_win_size + classical_lag,
        classical_win_size + classical_lag + len(classical_targets),
    )
    common_positions = np.intersect1d(qrc_positions, classical_positions)

    if common_positions.size == 0:
        min_len = min(len(qrc_targets), len(classical_targets), plot_limit)
        common_positions = np.arange(min_len)
        true_values = np.asarray(test_series[:min_len]).ravel()
        return {
            "x": common_positions,
            "true_values": true_values,
            "qrc_preds": np.asarray(qrc_preds).ravel()[:min_len],
            "classical_preds": np.asarray(classical_preds).ravel()[:min_len],
        }

    common_positions = common_positions[:plot_limit]
    qrc_indices = np.searchsorted(qrc_positions, common_positions)
    classical_indices = np.searchsorted(classical_positions, common_positions)

    return {
        "x": common_positions,
        "true_values": np.asarray(test_series[common_positions]).ravel(),
        "qrc_preds": np.asarray(qrc_preds).ravel()[qrc_indices],
        "classical_preds": np.asarray(classical_preds).ravel()[classical_indices],
    }


def plot_best_model_comparison(best_qrc_row, best_classical_row, data_profile_config, constants,
                               plot_style="color", show=True):
    """Plot the best QRC and classical ESN results against the target series."""
    plot_style = validate_plot_style(plot_style)
    style = get_plot_style_config(plot_style)
    profile_name = data_profile_config['name']
    print(f"\n{'='*60}\n--- Generating plot for profile: {profile_name} ---\n")

    # Regenerate the time series.
    generator_func = data_profile_config['generator']
    time_series = generator_func(**data_profile_config['params'])
    train_series, test_series, _ = split_and_scale_series(time_series, constants['TRAIN_FRACTION'])
    figure_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Retrain the best QRC model.
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

    # Retrain the best classical ESN for console comparison.
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

    plot_limit = 200
    aligned = _align_predictions_to_common_positions(
        test_series=test_series,
        qrc_win_size=qrc_win_size,
        qrc_lag=qrc_lag,
        qrc_targets=qrc_test_outputs,
        qrc_preds=qrc_preds,
        classical_win_size=classical_win_size,
        classical_lag=classical_lag,
        classical_targets=classical_test_outputs,
        classical_preds=classical_preds,
        plot_limit=plot_limit,
    )

    x_values = aligned["x"]
    true_values = aligned["true_values"]
    qrc_preds_plot = aligned["qrc_preds"]
    classical_preds_plot = aligned["classical_preds"]
    qrc_error = np.abs(true_values - qrc_preds_plot)
    classical_error = np.abs(true_values - classical_preds_plot)

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )

    ax_top.plot(x_values, true_values, label="True Data", **style["line_styles"]["truth"])
    ax_top.plot(x_values, qrc_preds_plot, label="Best QRC", **style["line_styles"]["qrc"])
    ax_top.plot(
        x_values,
        classical_preds_plot,
        label="Best Classical ESN",
        **style["line_styles"]["classical"],
    )
    ax_top.set_ylabel("Normalized Value", fontsize=12)
    ax_top.set_title(profile_name, fontsize=16, weight="bold")
    ax_top.legend(loc="upper right", fontsize=10, frameon=True)
    ax_top.grid(True, linestyle="--", alpha=style["grid_alpha"])

    annotation_text = "\n".join(
        [
            "QRC:",
            (
                f"MSE {best_qrc_row['median_mse']:.3e} | repro {qrc_mse:.3e} | "
                f"CV {best_qrc_row['cv_mse']:.2f}"
            ),
            (
                f"win {qrc_win_size}, layers {int(best_qrc_row['n_layers'])}, "
                f"leak {best_qrc_row['leakage_rate']:.2f}"
            ),
            (
                f"lambda {best_qrc_row['lambda_reg']:.2e}, lag {qrc_lag}, seed {qrc_seed}"
            ),
            "",
            "Classical ESN:",
            (
                f"MSE {best_classical_row['median_mse']:.3e} | repro {classical_mse:.3e} | "
                f"CV {best_classical_row['cv_mse']:.2f}"
            ),
            (
                f"win {classical_win_size}, res {_get_optional_int(best_classical_row, 'reservoir_size', 10)}, "
                f"leak {best_classical_row['leakage_rate']:.2f}"
            ),
            (
                f"radius {best_classical_row['spectral_radius']:.2f}, "
                f"sparsity {best_classical_row['sparsity']:.2f}, seed {classical_seed}"
            ),
        ]
    )
    ax_top.text(
        0.015,
        0.98,
        annotation_text,
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": style["annotation_facecolor"],
            "edgecolor": style["annotation_edgecolor"],
            "alpha": 0.95,
        },
    )

    ax_bottom.plot(x_values, qrc_error, label="QRC Absolute Error", **style["line_styles"]["qrc_error"])
    ax_bottom.plot(
        x_values,
        classical_error,
        label="Classical ESN Absolute Error",
        **style["line_styles"]["classical_error"],
    )
    ax_bottom.set_xlabel("Prediction Step in Test Set", fontsize=12)
    ax_bottom.set_ylabel("Absolute Error", fontsize=12)
    ax_bottom.grid(True, linestyle="--", alpha=style["grid_alpha"])
    ax_bottom.legend(loc="upper right", fontsize=10, frameon=True)

    fig.suptitle("One-Step-Ahead Prediction Comparison", fontsize=18, y=0.98)
    fig.tight_layout()
    fig.savefig(
        figure_dir / f"{profile_name}_comparison_{plot_style}.png",
        dpi=300,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close(fig)
