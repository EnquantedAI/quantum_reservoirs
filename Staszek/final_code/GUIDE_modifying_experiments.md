# Guide: Upgrading the QRC to ZZ / IQP Phase Encoding (`final_code`)

The participation-ratio-below-~1.5 puzzle is **not a charting bug**. The current encoder
(`models.py`) uses one `Rx` per qubit, so window values are encoded **independently**; since
consecutive series values are nearly identical they **homogenise** in Hilbert space → low
participation ratio no matter how much data. Fix: encode with a **ZZ / IQP feature map**
(phase + entanglement) on data scaled to **[0, π]**, and drive the model with a **grid search
over scalers** (weight, bias, phase) + leakage.

This guide is scoped to the `final_code/` package only.

## Layout (what each file does)

```
src/models.py            quantum_feature_map (THE circuit) + train_esn_reservoir / predict_esn
src/data_generation.py   raw generators + split_and_scale_series (min-max -> [0,1])
src/experiment.py        run_single_qrc_trial -> run_qrc_experiment_with_subseeds (11 sub-seeds,
                         median/std/CV); materialize_qrc_feature_channels (F_win/F_esn/F_joint)
src/analysis.py          get_qrc_feature_space, calculate_participation_ratio,
                         memory_capacity_spectrum, canonical_correlations
src/visualization.py     plot_best_model_comparison (saves to reports/figures/)
notebooks/1_run_experiments.ipynb   grid search -> data/results_comparative.csv
notebooks/02_analysis_and_plots.ipynb / 03_pca_analysis.ipynb / 04_memory_duality.ipynb
tests/test_submission_pipeline.py   unit tests (scaling, schema, PR, MC, CCA)
```

**Golden rule:** thread each new knob (`weight_scaler`, `bias_scaler`, `phase_scaler`, `iqp`)
through every place that builds features. In *this* package the circuit + random params are
materialised in **two** spots — `train_esn_reservoir` **and** `materialize_qrc_feature_channels`
— so both must be updated or `F_joint` will diverge from what the readout actually trained on.

---

## 1. `src/models.py` — replace the circuit

Old: `Rx` encoding → `Rot` (U) + linear `CNOT`. New: ZZ/IQP encoding → `Rot` (U) + `Rx`
reuploading + circular `CNOT`.

```python
def zz_feature_map(features, scaler=2.0, IQP=False):
    """scaler~2.0,IQP=False -> Havlicek ZZ ; scaler~1.0,IQP=True -> PennyLane IQP."""
    import pennylane as qml
    n = len(features)
    for i in range(n):
        qml.Hadamard(wires=i)
    for i in range(n):
        qml.RZ(scaler * features[i], wires=i)
    for i in range(n):
        for j in range(i + 1, n):
            angle = (scaler * features[i] * features[j] if IQP
                     else scaler * (np.pi - features[i]) * (np.pi - features[j]))
            qml.MultiRZ(angle, wires=[i, j])


def quantum_feature_map(inputs, weights, biases, n_layers, n_qubits, dev,
                        weight_scaler=1.0, bias_scaler=1.0, phase_scaler=2.0, iqp=False):
    import pennylane as qml

    @qml.qnode(dev)
    def circuit(inputs, weights, biases):
        zz_feature_map(inputs, scaler=phase_scaler, IQP=iqp)          # phase + entanglement
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*(weight_scaler * weights[layer, i]), wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])                        # circular
            if layer + 1 < n_layers:                                 # Rx reuploading
                for i in range(n_qubits):
                    qml.RX(inputs[i] + bias_scaler * biases[i], wires=i)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
               [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] + \
               [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit(inputs, weights, biases)
```

Then in **`train_esn_reservoir`**: add the four args to the signature, forward them to the
`quantum_feature_map(...)` call, and **scale the encoded input to [0, π]** — pass
`inputs=np.pi * classical_states[t]` (the ESN state is in [0,1]). The readout
(`n_observables = 3 * n_qubits`) is unchanged.

> `weight_scaler`/`bias_scaler` can be applied either when the random `weights`/`biases` are
> generated (`np.random.uniform(...) * weight_scaler`) **or** inside the circuit as above —
> pick one place and be consistent. The snippet applies them in the circuit, so leave the
> random generation untouched.

---

## 2. `src/experiment.py` — thread + log + fix the channels

* **`run_single_qrc_trial`**: the `params` tuple grows 5 → 9:
  `leakage_rate, lambda_reg, window_size, n_layers, lag, weight_scaler, bias_scaler, phase_scaler, iqp = params`.
  Pass the four scalers into both `train_esn_reservoir(...)` and `predict_esn(...)`.
* **`run_qrc_experiment_with_subseeds`**: unpack the same 9-tuple and add the four values to the
  returned dict (→ new CSV columns automatically).
* **`materialize_qrc_feature_channels`**: unpack the 9-tuple too, and pass the scalers + the
  `np.pi *` input scaling into its internal `quantum_feature_map(...)` call, so `F_joint`
  matches the new model.

---

## 3. `notebooks/1_run_experiments.ipynb` — extend the grid

Order **must** match the unpacking in step 2:

```python
param_grid_qrc = {
    'leakage_rate':  [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],  # refine low end (new optima ~0.07-0.2)
    'lambda_reg':    [1e-8],
    'window_size':   [2, 4, 6, 8, 10],   # >=2: ZZ needs a pair to entangle
    'n_layers':      [1, 2, 4],
    'lag':           [0],
    'weight_scaler': [0.2, 0.5, 1.0],    # NEW
    'bias_scaler':   [0.2, 0.5, 1.0],    # NEW
    'phase_scaler':  [0.5, 1.0, 2.0],    # NEW
    'iqp':           [True, False],      # NEW
}
```

Main loop and save cell need no change — they iterate the grid generically and write
`data/results_comparative.csv`.

---

## 4. Update the analysis params tuples

The new columns must flow into the analysis notebooks, which rebuild `params` tuples by hand:

* **`src/analysis.py → get_qrc_feature_space`**: unpack the 9-tuple, pass scalers to
  `train_esn_reservoir`.
* **`notebooks/03_pca_analysis.ipynb`**: extend `params_tuple` with the 4 new fields (read from
  the CSV row).
* **`notebooks/04_memory_duality.ipynb`**: the `params = (...)` tuples in `compute_mc_for_config`,
  the CCA loop, and the α-ablation must append the 4 scaler values.

---

## 5. Run, test, collect

```powershell
# from final_code/
python -m pytest tests/ -q          # or: python -m unittest -q  (sanity-check the edits)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 1_run_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_analysis_and_plots.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_pca_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_memory_duality.ipynb
```

* **Numbers:** `data/results_comparative.csv` (now incl. `weight_scaler`/`bias_scaler`/`phase_scaler`/`iqp`).
* **Tables:** inline in `02_analysis_and_plots.ipynb`.
* **Charts:** `reports/figures/` — comparison plots, MSE-vs-participation-ratio (03), and the
  memory-duality panels (MC decomposition / CCA / ablation, 04).
* The **participation ratio** (`calculate_participation_ratio`) is the metric in 03; with the
  new encoder it should finally climb above the old ~<2 ceiling.

⚠️ The grid is a full Cartesian product — the 4 new knobs multiply runtime heavily and ZZ adds
O(n²) gates per layer. **Test on a reduced grid first** (one profile, fewer scaler values).
Runs are already parallelised (`joblib`, `n_jobs=-1`).

---

## Notes for Jacob

- `models.py` here still ships the **old Rx** circuit — steps 1–4 swap in the ZZ/IQP model.
- `weights`/`biases` and the circuit are built in **two** places (`train_esn_reservoir` and
  `materialize_qrc_feature_channels`); both are updated so `F_joint` stays faithful.
- The `tests/` suite asserts the result-dict schema and the participation-ratio/MC/CCA maths —
  run it after editing; adding the 4 columns is additive and shouldn't break it.
