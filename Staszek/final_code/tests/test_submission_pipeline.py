import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import (
    calculate_effective_dimension,
    calculate_participation_ratio,
    canonical_correlations,
    memory_capacity_spectrum,
)
from src.data_generation import generate_arma_data, split_and_scale_series
from src.experiment import (
    run_classical_experiment_with_subseeds,
    run_qrc_experiment_with_subseeds,
)
from src.models import (
    initialize_classical_reservoir,
    predict_esn_classical,
    train_classical_reservoir,
)


class SubmissionPipelineTests(unittest.TestCase):
    def test_split_and_scale_uses_train_statistics_only(self):
        series = np.array([0.0, 1.0, 2.0, 10.0])

        train_scaled, test_scaled, scaler = split_and_scale_series(series, 0.75)

        self.assertTrue(np.allclose(train_scaled, np.array([0.0, 0.5, 1.0])))
        self.assertTrue(np.allclose(test_scaled, np.array([5.0])))
        self.assertEqual(scaler["min"], 0.0)
        self.assertEqual(scaler["max"], 2.0)

    def test_generators_return_raw_series(self):
        arma_series = generate_arma_data(n_points=100, seed=42)
        self.assertLess(float(np.min(arma_series)), 0.0)

    def test_classical_prediction_defaults_to_cold_start(self):
        train_inputs = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.3],
                [0.3, 0.4],
                [0.4, 0.5],
            ]
        )
        train_outputs = np.array([0.3, 0.4, 0.5, 0.6])
        test_inputs = np.array(
            [
                [0.5, 0.6],
                [0.6, 0.7],
            ]
        )
        reservoir_size = 6
        leakage_rate = 0.3

        W_in, W_res = initialize_classical_reservoir(
            reservoir_size=reservoir_size,
            input_dim=2,
            spectral_radius=0.9,
            sparsity=0.1,
            seed=2025,
        )
        W_out, final_state = train_classical_reservoir(
            train_inputs,
            train_outputs,
            W_in,
            W_res,
            reservoir_size=reservoir_size,
            leakage_rate=leakage_rate,
            lambda_reg=1e-8,
        )

        preds_default = predict_esn_classical(
            test_inputs,
            W_in,
            W_res,
            W_out,
            reservoir_size=reservoir_size,
            leakage_rate=leakage_rate,
        )
        preds_zero = predict_esn_classical(
            test_inputs,
            W_in,
            W_res,
            W_out,
            reservoir_size=reservoir_size,
            leakage_rate=leakage_rate,
            initial_state=np.zeros(reservoir_size),
        )
        preds_carry = predict_esn_classical(
            test_inputs,
            W_in,
            W_res,
            W_out,
            reservoir_size=reservoir_size,
            leakage_rate=leakage_rate,
            initial_state=final_state,
        )

        self.assertTrue(np.allclose(preds_default, preds_zero))
        self.assertFalse(np.allclose(preds_default, preds_carry))

    def test_experiment_wrappers_publish_new_schema_fields(self):
        time_series = generate_arma_data(n_points=80, seed=7)
        profile = {"name": "ARMA_1_2_stochastic"}

        qrc_result = run_qrc_experiment_with_subseeds(
            params=(0.3, 1e-8, 2, 1, 0),
            profile=profile,
            time_series=time_series,
            train_fraction=0.7,
            base_seed=2025,
            num_trials=3,
        )
        classical_result = run_classical_experiment_with_subseeds(
            params=(10, 0.9, 0.1, 0.3, 1e-8, 2),
            profile=profile,
            time_series=time_series,
            train_fraction=0.7,
            base_seed=2025,
            num_trials=3,
        )

        self.assertEqual(qrc_result["eval_protocol"], "cold_start")
        self.assertEqual(classical_result["eval_protocol"], "cold_start")
        self.assertEqual(qrc_result["num_trials"], 3)
        self.assertEqual(classical_result["num_trials"], 3)
        self.assertEqual(classical_result["window_size"], 2)
        self.assertEqual(classical_result["lag"], 0)
        self.assertIn("representative_seed", qrc_result)
        self.assertIn("representative_seed", classical_result)

    def test_degenerate_feature_space_returns_nan(self):
        feature_matrix = np.ones((4, 3))
        self.assertTrue(np.isnan(calculate_participation_ratio(feature_matrix)))
        self.assertTrue(np.isnan(calculate_effective_dimension(feature_matrix)))

    def test_participation_ratio_matches_expected_value(self):
        feature_matrix = np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
                [0.0, 1.0],
            ]
        )
        self.assertTrue(np.isclose(calculate_participation_ratio(feature_matrix), 2.0))


def _build_esn_states(inputs, leakage_rate, state_dim):
    states = np.zeros((len(inputs), state_dim))
    current = np.zeros(state_dim)
    for t, u in enumerate(inputs):
        current = (1 - leakage_rate) * current + leakage_rate * u
        states[t] = current
    return states


class MemoryCapacitySpectrumTests(unittest.TestCase):
    def test_window_features_have_sharp_cutoff_at_window_size(self):
        rng = np.random.default_rng(7)
        window_size = 4
        n_steps = 600
        series = rng.uniform(-1.0, 1.0, size=n_steps)

        n_samples = n_steps - window_size + 1
        features = np.stack(
            [series[t : t + window_size] for t in range(n_samples)]
        )
        reference = features[:, -1]

        lags, mc = memory_capacity_spectrum(
            features, reference, lags=range(0, 2 * window_size), lambda_reg=1e-10
        )

        for k in range(window_size):
            self.assertGreater(
                mc[k], 0.9,
                msg=f"lag {k} inside window should recover near-perfectly, got MC={mc[k]:.3f}",
            )
        for k in range(window_size, 2 * window_size):
            self.assertLess(
                mc[k], 0.1,
                msg=f"lag {k} outside window should drop to ~0, got MC={mc[k]:.3f}",
            )

    def test_esn_memory_decays_faster_for_higher_leakage(self):
        rng = np.random.default_rng(11)
        n_steps = 800
        state_dim = 6
        inputs = rng.uniform(-1.0, 1.0, size=(n_steps, 1))

        states_low = _build_esn_states(inputs, leakage_rate=0.2, state_dim=state_dim)
        states_high = _build_esn_states(inputs, leakage_rate=0.9, state_dim=state_dim)
        reference = inputs[:, 0]

        lags = list(range(0, 20))
        _, mc_low = memory_capacity_spectrum(
            states_low, reference, lags=lags, lambda_reg=1e-8
        )
        _, mc_high = memory_capacity_spectrum(
            states_high, reference, lags=lags, lambda_reg=1e-8
        )

        area_low = np.nansum(mc_low)
        area_high = np.nansum(mc_high)
        self.assertGreater(
            area_low, area_high,
            msg=f"low-leakage MC area ({area_low:.3f}) should exceed high-leakage area ({area_high:.3f})",
        )
        mid = 6
        self.assertGreater(
            mc_low[mid], mc_high[mid],
            msg=f"at mid lag {mid} low-leakage MC={mc_low[mid]:.3f} should exceed high-leakage MC={mc_high[mid]:.3f}",
        )


class CanonicalCorrelationTests(unittest.TestCase):
    def test_identical_inputs_yield_unit_correlations(self):
        rng = np.random.default_rng(3)
        F = rng.standard_normal(size=(200, 5))
        corrs = canonical_correlations(F, F.copy())
        self.assertEqual(corrs.shape[0], 5)
        self.assertTrue(
            np.allclose(corrs, 1.0, atol=1e-8),
            msg=f"expected all ones, got {corrs}",
        )

    def test_independent_inputs_yield_low_correlations(self):
        rng = np.random.default_rng(5)
        F_a = rng.standard_normal(size=(500, 3))
        F_b = rng.standard_normal(size=(500, 3))
        corrs = canonical_correlations(F_a, F_b)
        self.assertLess(
            corrs[0], 0.3,
            msg=f"independent Gaussian columns should give low top CCA, got {corrs}",
        )

    def test_partial_shared_subspace_recovered(self):
        rng = np.random.default_rng(17)
        shared = rng.standard_normal(size=(400, 1))
        F_a = np.hstack([shared, rng.standard_normal(size=(400, 2))])
        F_b = np.hstack(
            [
                shared + 0.01 * rng.standard_normal(size=(400, 1)),
                rng.standard_normal(size=(400, 2)),
            ]
        )
        corrs = canonical_correlations(F_a, F_b)
        self.assertGreater(corrs[0], 0.95)
        self.assertLess(corrs[-1], 0.3)


if __name__ == "__main__":
    unittest.main()
