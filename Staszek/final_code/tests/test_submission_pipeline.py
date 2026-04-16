import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import calculate_effective_dimension, calculate_participation_ratio
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


if __name__ == "__main__":
    unittest.main()
