import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

@qml.qnode(qml.device('lightning.qubit', wires=2))
def circuit(a, w):
    theta = 0
    i = 0
    j = 1
    qml.CNOT(wires=[i, j])
    qml.RZ(theta, wires=j)
    qml.CNOT(wires=[i, j])
    qml.RX(theta, wires=i)
    qml.RX(theta, wires=j)

print(qml.draw(circuit)(a=2.3, w=[1.2, 3.2, 0.7]))
