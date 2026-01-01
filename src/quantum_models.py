import pennylane as qml
from pennylane import numpy as np

# Set up a 4-qubit device using the default qubit simulator
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def angle_embedding_layer(inputs):
    """Encodes classical data using Angle Embedding (RY gates)."""
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

def strongly_entangling_layer(weights):
    """A variational layer with rotations and CNOT entanglers."""
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

@qml.qnode(dev)
def q_classifier_circuit(weights, inputs):
    """
    The full Variational Quantum Circuit (VQC).
    1. Embedding: Data -> Quantum State
    2. Ansatz: Parameterized gates (the 'model' weights)
    3. Measurement: Expectation value of the PauliZ operator
    """
    # Step 1: Data Embedding
    angle_embedding_layer(inputs)
    
    # Step 2: Variational Layers (Ansatz)
    strongly_entangling_layer(weights)
    
    # Step 3: Measurement (We measure qubit 0 to get a prediction)
    return qml.expval(qml.PauliZ(0))

def initialize_weights(n_layers=3):
    """Initializes random weights for the variational circuit."""
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return np.random.random(size=shape, requires_grad=True)

if __name__ == "__main__":
    # Test a single forward pass
    dummy_input = np.array([0.1, 0.2, 0.3, 0.4], requires_grad=False)
    weights = initialize_weights(n_layers=2)
    prediction = q_classifier_circuit(weights, dummy_input)
    print(f"Quantum Circuit Output (Expectation Value): {prediction}")