import numpy as np
import jax.random
import jax.numpy as jnp
import jax.scipy.optimize
import jaxopt
jax.config.update("jax_enable_x64", True)
from functools import partial
import os
import contextlib

from dla import purity
from loss import single_loss, loss, acc, pred

@partial(jax.jit, static_argnames=["cl", "nqubits", "loss_type", "gate_id"])
def sort_gs(
    w: jnp.ndarray,  # weights
    b: jnp.ndarray,  # bias
    gs: jnp.ndarray,  # ground states
    labels: jnp.ndarray,  # labels
    js: jnp.ndarray,  # j coordinates
    ascending: bool,  # ascending or descending order
    cl: str,  # curriculum learning type
    nqubits: int,  # number of qubits
    loss_type: str,  # loss type
    gate_id: str,  # gate identifier
    gen_basis: list[jnp.ndarray],  # generator basis
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sorts the ground states by their scores in ascending or descending order.

    Parameters:
    w (jnp.ndarray): Weights of the QCNN.
    b (jnp.ndarray): Bias of the QCNN.
    gs (jnp.ndarray): Ground states as a 2D numpy array.
    labels (jnp.ndarray): Labels corresponding to each ground state.
    js (jnp.ndarray): j coordinates of the ground states.
    ascending (bool): If True, the ground states are sorted in ascending order, otherwise in descending order.
    cl (str): The type of curriculum learning.
    nqubits (int): The number of qubits in the system.
    loss_type (str): The type of loss to compute. Must be either "cross_entropy" or "mean_squares".
    gate_id (str): The identifier for the type of gate sequence to apply.
    gen_basis (list[jnp.ndarray]): The generator basis for calculating the purity of a state with respect to the DLA of the circuit.
    
    Returns:
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: The sorted ground states, labels and j indices.
    """
    if cl in ["PCL", "PACL"]:
        scores = jax.vmap(purity, in_axes=[0, None])(gs, gen_basis)
    else:
        scores = jax.vmap(single_loss, in_axes=[None, None, 0, 0, None, None, None])(w, b, gs, labels, nqubits, loss_type, gate_id)

    p = jnp.where(ascending, scores.argsort(), scores.argsort()[::-1])
    
    return gs[p], labels[p], js[p]


def train_qcnn(
    gs_train: jnp.ndarray,
    gs_val: jnp.ndarray,
    labels_train: jnp.ndarray,
    labels_val: jnp.ndarray,
    j_train: jnp.ndarray,
    opt: jaxopt.OptaxSolver,
    cl: str,
    nqubits: int,
    num_iters: int,
    max_weight_init: float,
    nweights: int,
    initialization: str,
    with_bias: bool,
    with_val: bool,
    cl_pace: jnp.ndarray,
    batch_size: int,
    optimizer: str,
    loss_type: str,
    gate_id: str,
    gen_basis: list[jnp.ndarray],
) -> tuple[
    list[jnp.ndarray],
    list[jnp.ndarray],
    list[float],
    list[jnp.ndarray],
    list[jnp.ndarray],
    list[float],
    list[float],
    jnp.ndarray,
]:
    """
    Train a Quantum Convolutional Neural Network (QCNN) using the given parameters.

    Parameters:
    gs_train (jnp.ndarray): The ground states to be used for training.
    gs_val (jnp.ndarray): The ground states to be used for validation.
    labels_train (jnp.ndarray): The labels corresponding to the ground states used for training.
    labels_val (jnp.ndarray): The labels corresponding to the ground states used for validation.
    j_train (jnp.ndarray): The j coordinates of the ground states used for training.
    opt (jaxopt.OptaxSolver): The optimizer to be used.
    cl (str): The type of curriculum learning to be used.
    nqubits (int): The number of qubits in the system.
    num_iters (int): The number of iterations to be performed.
    max_weight_init (float): The maximum value of the weights to be used for initialization.
    nweights (int): The number of weights in the QCNN.
    initialization (str): The type of initialization to be used for the weights.
    with_bias (bool): Whether to use bias or not.
    with_val (bool): Whether to use validation or not.
    cl_pace (jnp.ndarray): The pace of the curriculum learning.
    batch_size (int): The size of the batch to be used for training.
    optimizer (str): The type of optimizer to be used.
    loss_type (str): Specifies the type of loss function to be used. Can be either "cross_entropy" or "mean_squares".
    gate_id (str): Specifies the identifier for the type of gate sequence to apply. Can be "general" or "a0".,
    gen_basis (list[jnp.ndarray]): The generator basis for calculating the purity of a state with respect to the DLA of the circuit.
    
    Returns:
    tuple[
        list[jnp.ndarray],
        list[jnp.ndarray],
        list[float],
        list[jnp.ndarray],
        list[jnp.ndarray],
        list[float],
        list[float],
        jnp.ndarray,
    ]: The weights, bias, losses, predictions for training, predictions for validation, accuracy for training, accuracy for validation, and the j coordinates of the ground states used for training.
    """

    # Initialize weights
    if initialization == "uniform":
        weights_init = np.random.uniform(0, max_weight_init, nweights)
    elif initialization == "gaussian":
        weights_init = np.random.normal(0, 1 / np.sqrt(nqubits), nweights)

    bias_init = np.array([0.0] * 4)

    if with_bias:
        params_init = [weights_init, bias_init]
    else:
        params_init = weights_init

    # Initialize variables
    weights = []
    bias = []
    losses = []
    pred_train_arr = []
    pred_val_arr = []
    acc_train_arr = []
    acc_val_arr = []

    w = weights_init
    b = bias_init

    params = params_init
    state = opt.init_state(params_init, gs_train[:2], labels_train[:2], nqubits, with_bias, loss_type, gate_id)

    for it in range(num_iters):
        # For self paced learning, we sort the datapoints at every iteration
        if cl in ["SPCL", "SPACL", "FSPACL", "FSPCL"]:
            ascending = True if cl in ["SPCL", "FSPCL"] else False
            gs_train, labels_train, j_train = sort_gs(
                w, b, gs_train, labels_train, j_train, ascending, cl, nqubits, loss_type, gate_id, gen_basis
            )

        # Once they are sorted, we select the first datapoints into the batch lists
        if cl in ["CL", "ACL", "PCL", "PACL", "RAND", "SPACL", "SPCL"]:
            batch_index = np.random.default_rng().choice(cl_pace[it], size=batch_size, replace=False)

            gs_train_batch = gs_train[:cl_pace[it]][batch_index]
            labels_train_batch = labels_train[:cl_pace[it]][batch_index]

        elif cl in ["FSPACL", "FSPCL"]:
            gs_train_batch = gs_train[:batch_size]
            labels_train_batch = labels_train[:batch_size]

        elif cl == "NCL":
            batch_index = np.random.default_rng().choice(len(labels_train), size=batch_size, replace=False)

            gs_train_batch = gs_train[batch_index]
            labels_train_batch = labels_train[batch_index]

        # Update the weights by one optimizer step
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            params, state = opt.update(params, state, gs_train_batch, labels_train_batch, nqubits, with_bias, loss_type, gate_id)

        if with_bias:
            w, b = params
        else:
            w = params

        if optimizer == "GradientDescent":
            l = loss(params, gs_train_batch, labels_train_batch, nqubits, with_bias, loss_type, gate_id)
        else:
            l = state.value

        # Compute predictions and accuracy on train and validation set
        pred_train = pred(w, b, gs_train, nqubits, gate_id)
        if with_val:
            pred_val = pred(w, b, gs_val, nqubits, gate_id) if len(labels_val) > 0 else None
        else:
            pred_val = np.array([0] * len(labels_val))

        acc_train = acc(pred_train, labels_train)
        if with_val:
            acc_val = acc(pred_val, labels_val) if len(labels_val) > 0 else 0
        else:
            acc_val = 0

        weights.append(w.tolist())
        bias.append(b.tolist())

        # Save data for later plotting
        pred_train_arr.append(pred_train.tolist())
        pred_val_arr.append(pred_val.tolist())
        acc_train_arr.append(float(acc_train))
        acc_val_arr.append(float(acc_val))
        losses.append(float(l))

    return (
        weights,
        bias,
        losses,
        pred_train_arr,
        pred_val_arr,
        acc_train_arr,
        acc_val_arr,
        j_train,
    )
