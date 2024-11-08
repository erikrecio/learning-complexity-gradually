import numpy as np
import jax.random
import jax.numpy as jnp
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)
from functools import partial

from qcnn import variational_classifier

#########################
######### Loss ##########
#########################

@partial(jax.jit, static_argnames=["nqubits", "loss_type", "gate_id"])
def single_loss(
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    ground_state: jnp.ndarray,
    label: int,
    nqubits: int,
    loss_type: str,
    gate_id: str
) -> float:
    """
    Computes the loss of a single ground state.

    Parameters:
    weights (jnp.ndarray): Weights of the QCNN.
    bias (jnp.ndarray): Bias of the QCNN.
    ground_state (jnp.ndarray): The ground state as a 1D numpy array.
    label (int): The label of the ground state.
    nqubits (int): The number of qubits in the system.
    loss_type (str): The type of loss to compute. Must be either "cross_entropy" or "mean_squares".
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    float: The computed loss.
    """
    
    proj = variational_classifier(weights, bias, ground_state, nqubits, gate_id)

    if loss_type == "cross_entropy":
        cost = -jnp.log2(proj[label])
    
    elif loss_type == "mean_squares":
        cost = 1 + jnp.linalg.norm(proj)**2 - 2*proj[label]
    
    return cost

@partial(jax.jit, static_argnames=["nqubits", "with_bias", "loss_type", "gate_id"])
def loss(
    weights_and_bias: tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray,
    ground_states: jnp.ndarray,
    labels: jnp.ndarray,
    nqubits: int,
    with_bias: bool,
    loss_type: str,
    gate_id: str
) -> float:
    """
    Computes the average loss over a set of ground states.

    Parameters:
    weights_and_bias (tuple[jnp.ndarray, jnp.ndarray] | jnp.ndarray): The weights and bias of the QCNN.
    ground_states (jnp.ndarray): The ground states as a 2D numpy array.
    labels (jnp.ndarray): The labels corresponding to each ground state.
    nqubits (int): The number of qubits in the system.
    with_bias (bool): Whether to include a bias term in the computation.
    loss_type (str): The type of loss to compute. Must be either "cross_entropy" or "mean_squares".
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    float: The average computed loss across all ground states.
    """
    
    # Split weights and bias if with_bias is True, otherwise use a zero bias
    if with_bias:
        weights, bias = weights_and_bias
    else:
        weights = weights_and_bias
        bias = jnp.zeros(4)
        
    # Vectorize single_loss computation over the batch of ground states and labels
    costs = jax.vmap(single_loss, in_axes=[None, None, 0, 0, None, None, None])(
        weights, bias, ground_states, labels, nqubits, loss_type, gate_id
    )
    
    # Return the average loss
    return costs.mean()

#############################
######### Accuracy ##########
#############################

@partial(jax.jit, static_argnames=["nqubits", "gate_id"])
def single_pred(
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    ground_state: jnp.ndarray,
    nqubits: int,
    gate_id: str,
) -> int:
    """
    Computes the prediction of the QCNN for a single ground state.

    Parameters:
    weights (jnp.ndarray): The weights of the QCNN.
    bias (jnp.ndarray): The bias of the QCNN.
    ground_state (jnp.ndarray): The ground state as a 1D numpy array.
    nqubits (int): The number of qubits in the system.
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    int: The label corresponding to the maximum projector.
    """
    projectors = variational_classifier(weights, bias, ground_state, nqubits, gate_id)
    return np.argmax(projectors)

@partial(jax.jit, static_argnames=["nqubits", "gate_id"])
def pred(
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    ground_states: jnp.ndarray,
    nqubits: int,
    gate_id: str
) -> jnp.ndarray:
    """
    Computes predictions for a batch of ground states using the QCNN model.

    Parameters:
    weights (jnp.ndarray): The weights of the QCNN.
    bias (jnp.ndarray): The bias of the QCNN.
    ground_states (jnp.ndarray): A batch of ground states as a 2D numpy array.
    nqubits (int): The number of qubits in the system.
    gate_id (str): The identifier for the type of gate sequence to apply.

    Returns:
    jnp.ndarray: An array of predicted labels for each ground state.
    """
    # Vectorize single_pred computation over the batch of ground states
    predictions = jax.vmap(single_pred, in_axes=[None, None, 0, None, None])(
        weights, bias, ground_states, nqubits, gate_id
    )
    return predictions

@jax.jit
def acc(
    predictions: jnp.ndarray, 
    labels: jnp.ndarray
) -> float:
    """
    Computes the accuracy of the predictions with respect to the labels.

    Parameters:
    predictions (jnp.ndarray): An array of predicted labels.
    labels (jnp.ndarray): An array of true labels.

    Returns:
    float: The accuracy of the predictions as a percentage.
    """
    return (predictions == labels).sum() * 100 / len(labels)
