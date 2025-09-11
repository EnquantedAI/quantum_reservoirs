# src/models.py

import pennylane as qml
from pennylane import numpy as np


def get_q_device(n_qubits):
    """
    Creates a PennyLane quantum device.

    Parameters
    ----------
    n_qubits : int
        The number of qubits for the device.

    Returns
    -------
    qml.Device
        A PennyLane device instance.
    """
    return qml.device("default.qubit", wires=n_qubits, shots=None)


def quantum_feature_map(inputs, weights, biases, n_layers, n_qubits, dev):
    """
    Defines the quantum circuit that acts as a feature map.

    Parameters
    ----------
    inputs : np.ndarray
        Input data for the circuit.
    weights : np.ndarray
        Trainable weights for the quantum layers.
    biases : np.ndarray
        Biases added to the input data.
    n_layers : int
        Number of quantum layers in the circuit.
    n_qubits : int
        Number of qubits in the circuit.
    dev : qml.Device
        The PennyLane device to run the circuit on.

    Returns
    -------
    np.ndarray
        The expectation values of the observables, serving as quantum features.
    """
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
    """
    Trains the quantum reservoir computer.

    This involves generating random circuit parameters, computing quantum features
    for the entire training set, and then training a linear readout layer
    using Ridge Regression.

    Parameters
    ----------
    train_inputs : np.ndarray
        Training input data.
    train_outputs : np.ndarray
        Training target data.
    n_layers : int
        Number of layers in the quantum feature map.
    n_qubits : int
        Number of qubits to use.
    leakage_rate : float
        The leakage rate (alpha) of the reservoir.
    lambda_reg : float
        The regularization parameter for Ridge Regression.
    seed : int
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    tuple
        A tuple containing:
        - W_out (np.ndarray): The trained weights of the readout layer.
        - weights (np.ndarray): The randomly generated weights for the quantum circuit.
        - biases (np.ndarray): The randomly generated biases for the quantum circuit.
        - quantum_features (np.ndarray): The matrix of quantum features for analysis.
    """
    np.random.seed(seed)
    weights = np.random.uniform(-np.pi, np.pi, (n_layers, n_qubits, 3))
    biases = np.random.uniform(-0.5, 0.5, n_qubits)

    n_observables = 3 * n_qubits
    n_samples = len(train_inputs)
    dev = get_q_device(n_qubits)

    # 1. Compute classical ESN states
    classical_states = np.zeros((n_samples, n_qubits))
    current_classical_state = np.zeros(n_qubits)
    for t in range(n_samples):
        current_classical_state = (1 - leakage_rate) * current_classical_state + leakage_rate * train_inputs[t]
        classical_states[t] = current_classical_state

    # 2. Map classical states to quantum features
    quantum_features = np.zeros((n_samples, n_observables))
    for t in range(n_samples):
        quantum_features[t] = quantum_feature_map(
            inputs=classical_states[t], weights=weights, biases=biases,
            n_layers=n_layers, n_qubits=n_qubits, dev=dev
        )

    # 3. Train the readout layer (Ridge Regression)
    R = quantum_features
    Y = train_outputs.reshape(-1, 1)
    I = np.identity(n_observables)
    W_out = np.linalg.solve(R.T @ R + lambda_reg * I, R.T @ Y).flatten()

    return W_out, weights, biases, quantum_features


def predict_esn(test_inputs, weights, biases, W_out, n_layers, n_qubits, leakage_rate):
    """
    Makes one-step-ahead predictions using the trained QRC-ESN model.

    Parameters
    ----------
    test_inputs : np.ndarray
        The input data for which to make predictions.
    weights : np.ndarray
        The weights for the quantum circuit, obtained from training.
    biases : np.ndarray
        The biases for the quantum circuit, obtained from training.
    W_out : np.ndarray
        The trained readout weights.
    n_layers : int
        Number of layers in the quantum feature map.
    n_qubits : int
        Number of qubits.
    leakage_rate : float
        The leakage rate of the reservoir.

    Returns
    -------
    np.ndarray
        An array containing the model's predictions.
    """
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



# --- Classical ESN Model ---

def initialize_classical_reservoir(reservoir_size, input_dim, spectral_radius=0.9, sparsity=0.1, seed=2025):
    """
    Initializes the weight matrices for a classical Echo State Network.

    Parameters
    ----------
    reservoir_size : int
        The number of neurons in the reservoir.
    input_dim : int
        The dimensionality of the input signal (e.g., window_size).
    spectral_radius : float, optional
        The spectral radius of the reservoir weight matrix, by default 0.9.
    sparsity : float, optional
        The fraction of connections to set to zero in the reservoir matrix, by default 0.1.
    seed : int, optional
        Seed for the random number generator, by default 2025.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the input weight matrix (W_in) and the reservoir weight matrix (W_res).
    """
    np.random.seed(seed)

    W_in = np.random.uniform(-0.1, 0.1, (reservoir_size, input_dim))

    W_res = np.random.randn(reservoir_size, reservoir_size)
    W_res[np.random.rand(*W_res.shape) < sparsity] = 0

    eigenvalues = np.linalg.eigvals(W_res)
    current_spectral_radius = np.max(np.abs(eigenvalues))
    if current_spectral_radius > 1e-9:  # Avoid division by zero
        W_res = (spectral_radius / current_spectral_radius) * W_res

    return W_in, W_res


def update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate):
    """
    Updates the state of the classical reservoir for a single time step.

    Parameters
    ----------
    input_seq : np.ndarray
        The input vector at the current time step.
    W_in : np.ndarray
        The input weight matrix.
    W_res : np.ndarray
        The reservoir weight matrix.
    reservoir_state : np.ndarray
        The current state of the reservoir.
    leakage_rate : float
        The leakage rate (alpha) of the reservoir.

    Returns
    -------
    np.ndarray
        The updated reservoir state.
    """
    return (1 - leakage_rate) * reservoir_state + \
           leakage_rate * np.tanh(W_in @ input_seq + W_res @ reservoir_state)


def train_classical_reservoir(train_inputs, train_outputs, W_in, W_res, reservoir_size, leakage_rate, lambda_reg=1e-6):
    """
    Trains the readout layer of the classical ESN.

    Parameters
    ----------
    train_inputs : np.ndarray
        The sequence of training input vectors.
    train_outputs : np.ndarray
        The sequence of training target values.
    W_in : np.ndarray
        The input weight matrix.
    W_res : np.ndarray
        The reservoir weight matrix.
    reservoir_size : int
        The number of neurons in the reservoir.
    leakage_rate : float
        The leakage rate of the reservoir.
    lambda_reg : float, optional
        Regularization parameter for Ridge Regression, by default 1e-6.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the trained output weights (W_out) and the final reservoir state.
    """
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


def predict_esn_classical(test_inputs, W_in, W_res, W_out, reservoir_size, leakage_rate, initial_state):
    """
    Makes one-step-ahead predictions with the trained classical ESN.

    Parameters
    ----------
    test_inputs : np.ndarray
        The sequence of test input vectors.
    W_in : np.ndarray
        The input weight matrix.
    W_res : np.ndarray
        The reservoir weight matrix.
    W_out : np.ndarray
        The trained readout weight matrix.
    reservoir_size : int
        The number of neurons in the reservoir.
    leakage_rate : float
        The leakage rate of the reservoir.
    initial_state : np.ndarray
        The final state of the reservoir after training, used to initialize prediction.

    Returns
    -------
    np.ndarray
        An array containing the model's predictions.
    """
    predictions = []
    reservoir_state = initial_state.copy()

    for input_seq in test_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in, W_res, reservoir_state, leakage_rate)
        y_pred = (W_out @ reservoir_state)[0]
        predictions.append(y_pred)

    return np.array(predictions)


def get_classical_reservoir_states(train_inputs, reservoir_size, leakage_rate, spectral_radius, sparsity, seed, input_dim):
    """
    Drives the classical reservoir with training data and returns its internal states.

    Parameters
    ----------
    train_inputs : np.ndarray
        The sequence of training input vectors.
    reservoir_size, leakage_rate, spectral_radius, sparsity, seed, input_dim :
        Parameters needed to initialize and run the reservoir.

    Returns
    -------
    np.ndarray
        A matrix where each row is the reservoir state at a given time step.
    """
    W_in_c, W_res_c = initialize_classical_reservoir(reservoir_size, input_dim, spectral_radius, sparsity, seed)
    reservoir_states = []
    reservoir_state = np.zeros(reservoir_size)

    for input_seq in train_inputs:
        reservoir_state = update_reservoir_state(input_seq, W_in_c, W_res_c, reservoir_state, leakage_rate)
        reservoir_states.append(reservoir_state)

    return np.vstack(reservoir_states)