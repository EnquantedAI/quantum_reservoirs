import numpy as np


def get_q_device(n_qubits):
    """Create a PennyLane device."""
    import pennylane as qml

    return qml.device("default.qubit", wires=n_qubits, shots=None)


def quantum_feature_map(inputs, weights, biases, n_layers, n_qubits, dev):
    """Map one input vector to quantum features."""
    import pennylane as qml

    @qml.qnode(dev)
    def circuit(inputs, weights, biases):
        for i in range(n_qubits):
            total_angle = inputs[i] + biases[i]
            qml.RX(total_angle, wires=i)

        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        observables = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)] + \
                      [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] + \
                      [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        return observables

    return circuit(inputs, weights, biases)


def train_esn_reservoir(train_inputs, train_outputs, n_layers, n_qubits, leakage_rate, lambda_reg, seed):
    """Train the QRC readout and return its learned components."""
    np.random.seed(seed)
    weights = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    biases = np.random.uniform(-0.5, 0.5, n_qubits)

    n_observables = 3 * n_qubits
    n_samples = len(train_inputs)
    dev = get_q_device(n_qubits)

    # Compute classical reservoir states.
    classical_states = np.zeros((n_samples, n_qubits))
    current_classical_state = np.zeros(n_qubits)
    for t in range(n_samples):
        current_classical_state = (1 - leakage_rate) * current_classical_state + leakage_rate * train_inputs[t]
        classical_states[t] = current_classical_state

    # Map states to quantum features.
    quantum_features = np.zeros((n_samples, n_observables))
    for t in range(n_samples):
        quantum_features[t] = quantum_feature_map(
            inputs=classical_states[t], weights=weights, biases=biases,
            n_layers=n_layers, n_qubits=n_qubits, dev=dev
        )

    # Fit the readout layer.
    R = quantum_features
    Y = train_outputs.reshape(-1, 1)
    I = np.identity(n_observables)
    W_out = np.linalg.solve(R.T @ R + lambda_reg * I, R.T @ Y).flatten()

    return W_out, weights, biases, quantum_features


def predict_esn(test_inputs, weights, biases, W_out, n_layers, n_qubits, leakage_rate):
    """Predict with the trained QRC model."""
    predictions = []
    current_classical_state = np.zeros(n_qubits)
    dev = get_q_device(n_qubits)

    for input_val in test_inputs:
        current_classical_state = (1 - leakage_rate) * current_classical_state + leakage_rate * input_val
        q_features = quantum_feature_map(
            inputs=current_classical_state, weights=weights, biases=biases,
            n_layers=n_layers, n_qubits=n_qubits, dev=dev)
        y_pred = np.dot(W_out, q_features)
        predictions.append(y_pred)

    return np.array(predictions)



# Classical ESN helpers.

def initialize_classical_reservoir(reservoir_size, input_dim, spectral_radius=0.9, sparsity=0.1, seed=2025):
    """Initialize a classical ESN reservoir."""
    np.random.seed(seed)

    W_in = np.random.uniform(-0.1, 0.1, (reservoir_size, input_dim))

    W_res = np.random.randn(reservoir_size, reservoir_size)
    W_res[np.random.rand(*W_res.shape) < sparsity] = 0

    eigenvalues = np.linalg.eigvals(W_res)
    current_spectral_radius = np.max(np.abs(eigenvalues))
    if current_spectral_radius > 1e-9:  # Guard against division by zero.
        W_res = (spectral_radius / current_spectral_radius) * W_res

    return W_in, W_res


def update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate):
    """Advance the classical reservoir by one step."""
    return (1 - leakage_rate) * reservoir_state + \
           leakage_rate * np.tanh(W_in @ input_seq + W_res @ reservoir_state)


def train_classical_reservoir(train_inputs, train_outputs, W_in, W_res, reservoir_size, leakage_rate, lambda_reg=1e-6):
    """Train the classical ESN readout."""
    reservoir_states = []
    reservoir_state = np.zeros(reservoir_size)

    for input_seq in train_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate)
        reservoir_states.append(reservoir_state)

    R = np.vstack(reservoir_states)
    Y = train_outputs.reshape(-1, 1)
    I = np.identity(reservoir_size)
    W_out = np.linalg.solve(R.T @ R + lambda_reg * I, R.T @ Y).T

    return W_out, reservoir_state


def predict_esn_classical(test_inputs, W_in, W_res, W_out, reservoir_size, leakage_rate, initial_state=None):
    """Predict with the trained classical ESN."""
    predictions = []
    if initial_state is None:
        reservoir_state = np.zeros(reservoir_size)
    else:
        reservoir_state = initial_state.copy()

    for input_seq in test_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate)
        y_pred = (W_out @ reservoir_state)[0]
        predictions.append(y_pred)

    return np.array(predictions)


def get_classical_reservoir_states(train_inputs, reservoir_size, leakage_rate, spectral_radius, sparsity, seed, input_dim):
    """Return classical reservoir states for the training inputs."""
    W_in_c, W_res_c = initialize_classical_reservoir(reservoir_size, input_dim, spectral_radius, sparsity, seed)
    reservoir_states = []
    reservoir_state = np.zeros(reservoir_size)

    for input_seq in train_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in_c, W_res_c, reservoir_state, leakage_rate)
        reservoir_states.append(reservoir_state)

    return np.vstack(reservoir_states)
