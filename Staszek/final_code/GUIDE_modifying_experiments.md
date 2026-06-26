# Guide: Upgrading the QRC to a Phase-Encoding (ZZ / IQP) Model

The effective-dimension-below-2 issue is **not a charting bug** — the old `Rx` encoder treats
window values as independent, so near-identical time-series values homogenise in Hilbert space.
Fix: encode with a **ZZ / IQP feature map** (phase + entanglement) on data scaled to **[0, π]**,
and drive the model via a **grid search over scalers** (weight, bias, phase) + leakage rate.

---

## What to change (4 files)

**1. `src/models.py` — rewrite `quantum_feature_map`** (the circuit).
Old: `Rx` encoding → `Rot` (U) + linear `CNOT`. New: ZZ/IQP encoding → `Rot` (U) + `Rx`
reuploading + circular `CNOT`. *(If you have Jacob's attached ZZ implementation, drop it in —
it is authoritative; this is a faithful PennyLane equivalent.)*

```python
def quantum_feature_map(inputs, weights, biases, n_layers, n_qubits, dev,
                        weight_scaler=1.0, bias_scaler=1.0, phase_scaler=2.0, iqp=False):
    # phase_scaler~2.0 + iqp=False -> Havlicek ZZ ;  ~1.0 + iqp=True -> PennyLane IQP
    # phase_scaler<1 + iqp=True    -> ~old Rx behaviour.  inputs must be in [0, pi].
    @qml.qnode(dev)
    def circuit(inputs, weights, biases):
        # --- ZZ / IQP encoding (entangles + phase-encodes the window) ---
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(phase_scaler * inputs[i], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                phi = (phase_scaler * inputs[i] * inputs[j] if iqp
                       else phase_scaler * (np.pi - inputs[i]) * (np.pi - inputs[j]))
                qml.CNOT(wires=[i, j]); qml.RZ(phi, wires=j); qml.CNOT(wires=[i, j])
        # --- layers: U rotation + Rx reuploading + circular entanglement ---
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*(weight_scaler * weights[layer, i]), wires=i)
            for i in range(n_qubits):
                qml.RX(phase_scaler * inputs[i] + bias_scaler * biases[i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
               [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] + \
               [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit(inputs, weights, biases)
```

Then in the same file add `weight_scaler, bias_scaler, phase_scaler, iqp` to the signatures of
**`train_esn_reservoir`** and **`predict_esn`**, forward them to every `quantum_feature_map(...)`
call, and **scale the input to [0, π]**: pass `inputs=np.pi * classical_state` (so [0,1] → [0,π]).

**2. `src/experiment.py`** — in `run_single_qrc_trial` the QRC `params` tuple grows to 9:
`(leakage_rate, lambda_reg, window_size, n_layers, lag, weight_scaler, bias_scaler, phase_scaler, iqp)`.
Unpack it, pass the scalers down, and add the 4 new keys to the result dict in
`run_qrc_experiment_with_subseeds` (this creates the new CSV columns automatically).

**3. `notebooks/1_run_experiments.ipynb`** — extend `param_grid_qrc` (**order must match the tuple**):

```python
param_grid_qrc = {
    'leakage_rate':  [0.0, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],  # finer near low end
    'lambda_reg':    [1e-8],
    'window_size':   [2, 4, 6, 8, 10],   # >=2: ZZ needs a pair to entangle
    'n_layers':      [1, 2, 4],
    'lag':           [0],
    'weight_scaler': [0.5, 1.0, 1.5],
    'bias_scaler':   [0.5, 1.0, 1.5],
    'phase_scaler':  [0.5, 1.0, 2.0],
    'iqp':           [True, False],
}
```

**4. `src/analysis.py` + `notebooks/03_pca_analysis.ipynb`** — unpack the same 9-tuple in
`get_qrc_feature_space` and build `params_tuple` with the 4 new fields, so the PCA charts use
the new encoder. (This is where the effective dimension should finally climb above 2.)

---

## Run & collect

```powershell
New-Item -ItemType Directory -Force final_code/reports/figures   # plots crash without this
Copy-Item final_code/data/results_comparative.csv final_code/data/results_OLD.csv  # back up
```

Then run the notebooks in order: **1 → 2 → 3** (`jupyter nbconvert --to notebook --execute --inplace <nb>`).

- Numbers: `data/results_comparative.csv` (now includes the 4 new columns)
- Charts: `reports/figures/*.png`

⚠️ The grid is a full Cartesian product — the 4 new knobs multiply QRC runtime ~54× and the ZZ
map adds deeper circuits. **Test on a reduced grid first** (one profile, fewer scaler values).

---

## Notes

- Your ZZ implementation ("as attached") isn't in the repo — only the old-vs-new PDF is. Send it
  and I'll swap it in so the phase/IQP constants match yours exactly.
- I placed `bias_scaler·bias` on the reuploading `Rx` — a design choice; tell me if yours differs.
