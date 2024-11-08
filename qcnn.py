import numpy as np
import jax.random
import jax.numpy as jnp
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)
import pennylane as qml
from functools import partial

def convolutional_layer(q1: int, q2: int, weights: list[float], gate_id: str) -> None:
    """
    Applies a convolutional layer with specified gates based on the gate_id.

    Parameters:
    q1 (int): The first qubit wire.
    q2 (int): The second qubit wire.
    weights (list[float]): A list of weights for parameterized gates.
    gate_id (str): Identifier for the type of gate sequence to apply.
    """
    if gate_id == "general":
        # Apply parameterized U3 gates to qubit q1
        qml.U3(wires=q1, theta=weights[0], phi=weights[1], delta=weights[2])
        qml.U3(wires=q1, theta=weights[3], phi=weights[4], delta=weights[5])
        # Apply CNOT and rotation gates
        qml.CNOT(wires=[q2, q1])
        qml.RZ(wires=q1, phi=weights[6])
        qml.RY(wires=q2, phi=weights[7])
        qml.CNOT(wires=[q1, q2])
        qml.RY(wires=q2, phi=weights[8])
        qml.CNOT(wires=[q2, q1])
        # Apply additional U3 gates to qubit q1
        qml.U3(wires=q1, theta=weights[9], phi=weights[10], delta=weights[11])
        qml.U3(wires=q1, theta=weights[12], phi=weights[13], delta=weights[14])
        
    elif gate_id == "a0":
        # Apply RZ rotations and IsingXX interaction
        qml.RZ(wires=q1, phi=weights[0])
        qml.RZ(wires=q2, phi=weights[1])
        qml.IsingXX(wires=[q1, q2], phi=weights[2])
        qml.RZ(wires=q1, phi=weights[3])
        qml.RZ(wires=q2, phi=weights[4])
        
def pooling_layer(q1: int, q2: int, weights: list[float], gate_id: str) -> None:
    """
    Applies a pooling layer with specified gates based on the gate_id.

    Parameters:
    q1 (int): The first qubit wire.
    q2 (int): The second qubit wire.
    weights (list[float]): A list of weights for parameterized gates.
    gate_id (str): Identifier for the type of gate sequence to apply.
    """
    if gate_id == "general":
        # Apply parameterized U3 gates to qubit q1
        qml.U3(wires=q1, theta=weights[0], phi=weights[1], delta=weights[2])
        qml.U3(wires=q1, theta=weights[3], phi=weights[4], delta=weights[5])
        # Apply CNOT and rotation gates
        qml.CNOT(wires=[q2, q1])
        qml.RZ(wires=q1, phi=weights[6])
        qml.RY(wires=q2, phi=weights[7])
        qml.CNOT(wires=[q1, q2])
        qml.RY(wires=q2, phi=weights[8])
        qml.CNOT(wires=[q2, q1])
        # Apply additional U3 gates to qubit q1
        qml.U3(wires=q1, theta=weights[9], phi=weights[10], delta=weights[11])
        qml.U3(wires=q1, theta=weights[12], phi=weights[13], delta=weights[14])
    elif gate_id == "a0":
        # Apply RZ rotations and IsingXX interaction
        qml.RZ(wires=q1, phi=weights[0])
        qml.RZ(wires=q2, phi=weights[1])
        qml.IsingXX(wires=[q1, q2], phi=weights[2])
        qml.RZ(wires=q1, phi=weights[3])
        qml.RZ(wires=q2, phi=weights[4])

def cnn_circuit(
    weights: np.ndarray, state_ini: np.ndarray, nqubits: int, gate_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantum circuit for the convolutional neural network.

    Parameters:
    weights (np.ndarray): Weights for the parameterized gates.
    state_ini (np.ndarray): The initial quantum state.
    nqubits (int): The number of qubits in the quantum circuit.
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    tuple[np.ndarray, np.ndarray, np.ndarray]: The expectation values of Z0, Z1, and ZZ01.
    """
    qubits = list(range(nqubits))
    layers = int(np.log2(nqubits))
    gate_nweights = 15 if gate_id == "general" else 5
        
    qml.QubitStateVector(state_ini, wires=qubits)

    for j in range(layers-1):
        # Apply convolutional and pooling layers
        len_qubits = len(qubits)
        
        for i in range(len_qubits//2):
            convolutional_layer(
                qubits[2*i], qubits[(2*i+1)%len_qubits], weights[gate_nweights*2*j:gate_nweights*(2*j+1)], gate_id
            )
        
        for i in range(len_qubits//2):
            convolutional_layer(
                qubits[2*i+1], qubits[(2*i+2)%len_qubits], weights[gate_nweights*2*j:gate_nweights*(2*j+1)], gate_id
            )
            
        for i in range(len_qubits//2):
            pooling_layer(
                qubits[2*i], qubits[(2*i+1)%len_qubits], weights[gate_nweights*(2*j+1):gate_nweights*(2*j+2)], gate_id
            )

        qub = []
        for i in range(len_qubits):
            if i%2 == 1:
                qub.append(qubits[i])
                
        qubits = qub
    
    # Apply the final convolutional layer
    convolutional_layer(qubits[0], qubits[1], weights[gate_nweights*(2*layers-2):gate_nweights*(2*layers-1)], gate_id)
    
    return (
        qml.expval(qml.Z(qubits[0])),
        qml.expval(qml.Z(qubits[1])),
        qml.expval(qml.Z(qubits[0]) @ qml.Z(qubits[1]))
    )

def cnn(
    weights: jnp.ndarray,
    state_ini: jnp.ndarray,
    nqubits: int,
    gate_id: str,
) -> jnp.ndarray:
    """
    Applies a Quantum Convolutional Neural Network (QCNN) to the input quantum state.

    Parameters:
    weights (jnp.ndarray): The weights of the QCNN.
    state_ini (jnp.ndarray): The initial quantum state.
    nqubits (int): The number of qubits in the quantum state.
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    jnp.ndarray: A 1D numpy array containing the probabilities of the four possible outcomes.
    """
    
    dev = qml.device("default.qubit", wires=nqubits)
    cnn_circuit_node = qml.QNode(cnn_circuit, dev, interface="jax", diff_method="best")
    
    z0, z1, zz01 = cnn_circuit_node(weights, state_ini, nqubits, gate_id)

    # Compute the probabilities of the four possible outcomes
    proj_00 = (1+zz01+z0+z1)/4
    proj_01 = (1-zz01-z0+z1)/4
    proj_10 = (1-zz01+z0-z1)/4
    proj_11 = (1+zz01-z0-z1)/4

    return jnp.array([proj_00, proj_01, proj_10, proj_11])

@partial(jax.jit, static_argnames=["nqubits", "gate_id"])
def variational_classifier(
    weights: jnp.ndarray, bias: jnp.ndarray, state_ini: jnp.ndarray, nqubits: int, gate_id: str
) -> jnp.ndarray:
    """
    Applies a Variational Quantum Classifier (VQC) to the input quantum state.

    Parameters:
    weights (jnp.ndarray): The weights of the VQC.
    bias (jnp.ndarray): The bias of the VQC.
    state_ini (jnp.ndarray): The initial quantum state.
    nqubits (int): The number of qubits in the quantum state.
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    jnp.ndarray: A 1D numpy array containing the probabilities of the four possible outcomes.
    """
    return cnn(weights, state_ini, nqubits, gate_id) + bias
