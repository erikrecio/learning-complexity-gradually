import numpy as np
import jax.random
import jax.scipy.optimize
import jaxopt
import optax
jax.config.update("jax_enable_x64", True)
import time
from datetime import datetime
import os
import pytz

from dla import get_gen_basis
from save_data import save_data, save_hyperparameters
from training import train_qcnn, sort_gs
from ground_states import generate_gs
from loss import loss

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#-------------------------------- START SETTINGS --------------------------------------#

# "NCL" - No curriculum
# "CL" - Curriculum
# "ACL" - Anti-curriculum
# "RAND" - Random sorting
# "SPCL" - Self paced curriculum
# "SPACL" - Self paced anti-curriculum
# "PCL" - Physics inspired curriculum
# "PACL" - Physics inspired anti-curriculum
# "FSPCL" - Fixed Self paced curriculum
# "FSPACL" - Fixed Self paced anti-curriculum

# Running parameters
num_iters = 300    # Number of training iterations
num_runs = 5      # Number of training runs used to calculate the average loss and accuracy
cl_types =  ["NCL", "SPCL", "SPACL", "FSPACL"]
with_val = True

# Circuit and optimization parameters
nqubits = 16         # Num qubits, min 4, always 2**num_layers qubits
gate_id = "general"      # QCNN type, either "general", "a0"
with_bias = False    # Add a bias to the output of the quantum circuit
optimizer = "Adam"  # "Adam", "GradientDescent", "BFGS"
loss_type = "mean_squares" # "cross_entropy", "mean_squares"
initialization = "gaussian" # "gaussian", "uniform"

# Data hyper-parameters
ham = "gch" # "gch" (Generalized Cluster Hamiltonian) or "ssh" (Su-Schrieffer-Heeger)
batch_size = 10     # batch training size
train_size = 50      # Total ground states that will be used for training
val_size = 100      # Total gound states with training + validation

cl_iter_ratios = [1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20]

# Example of logarithmic pace
# cl_pace_ratios = np.array([10,	9,	5,	4,	3,	2,	2,	2,	2,	1,	2,	1,	1,	1,	1,	1,	0,	1,	1,	1])/50

# Example of linear pace
# cl_pace_ratios = np.array([10,	2,	2,	2,	2,	2,	2,	2,	2,	2,	3,	2,	2,	2,	2,	2,	2,	2,	2,	3])/50

# Example of exponential pace
cl_pace_ratios = np.array([10,	0,	1,	1,	2,	1,	1,	2,	1,	2,	2,	2,	2,	3,	2,	3,	3,	4,	3,	5])/50

gen_type = "a0"

# How the training data is generated
uniform_train = False    # True - Uniform, False - Balanced
uniform_val = False
epsilon_train = False   # True - epsilon, False - no epsilon
epsilon_val = False

# Tweak training hyper-parameters
max_weight_init = 2*np.pi  # weight_init goes from 0 to this number. Max = 2*np.pi. Other options = 0.01
stepsize = 0.01         # stepsize of the gradient descent.
file_name = ''

#-------------------------------- END SETTINGS --------------------------------------#

# Constant definitions
layers = int(np.log2(nqubits))
gate_nweights = 15 if gate_id == "general" else 5
nweights = 2*gate_nweights*(layers-1) + gate_nweights

cl_pace = []
i_batch_size = 0
for i in range(len(cl_iter_ratios)):
    if i < len(cl_iter_ratios)-1:
        i_batch_size += int(cl_pace_ratios[i]*train_size)
        i_num_iters = int(cl_iter_ratios[i]*num_iters)
    else:
        i_batch_size = train_size
        i_num_iters = num_iters - len(cl_pace)
        
    cl_pace += [i_batch_size]*i_num_iters

if cl_pace[0] < batch_size:
    raise Exception(f"CL starts with {cl_pace[0]} samples but the batch is larger with {batch_size} samples")


# Calculate the DLA basis if sorting by purity
gen_basis = []
if "PCL" in cl_types or "PACL" in cl_types:
    gen_basis = get_gen_basis(gen_type, nqubits)

time_now = datetime.now(pytz.timezone('Europe/Andorra')).strftime("%Y-%m-%d %H-%M-%S")

folder_name = f"Results/ham {ham} - {nqubits}q - {num_iters} iters/"
if not os.path.isdir(f'{folder_name}'):
    os.makedirs(f'{folder_name}')

save_hyperparameters(
    time_now,
    folder_name,
    file_name,
    num_iters,
    num_runs,
    cl_types,
    with_val,
    nqubits,
    with_bias,
    optimizer,
    loss_type,
    initialization,
    ham,
    batch_size,
    train_size,
    val_size,
    cl_pace_ratios,
    cl_iter_ratios,
    gen_type,
    uniform_train,
    uniform_val,
    epsilon_train,
    epsilon_val,
    max_weight_init,
    stepsize,
    gate_id,
)

# choose variational classifier
if optimizer == "GradientDescent":
    opt = jaxopt.GradientDescent(loss, stepsize=stepsize, verbose=False, jit=True)
elif optimizer == "Adam":
    opt = jaxopt.OptaxSolver(loss, optax.adam(stepsize), verbose=False, jit=False)
elif optimizer == "BFGS":
    opt = jaxopt.BFGS(loss, verbose=False, jit=True)


if with_val:
    print("Generating validation ground states...")
    start_time = time.time()
    gs_val, labels_val, j_val = generate_gs(val_size, uniform_val, epsilon_val, nqubits, ham)
    run_time = time.time() - start_time
    print(f"Validation ground states generated - {run_time:.0f}s")
    print()

for run in range (num_runs):
    
    # -------------------------------------------------------------- #
    # ------------------- Generate ground states ------------------- #
    # -------------------------------------------------------------- #
    
    print("Generating ground states...")
    start_time = time.time()
    gs_train, labels_train, j_train = generate_gs(train_size, uniform_train, epsilon_train, nqubits, ham)
    run_time = time.time() - start_time

    print(f"Ground states generated - {run_time:.0f}s")
    print()
    print("Max train / Last run")
    print("----------------------------------------------------")
    print("  CL    | Run |   Iter    |Acc train|Acc val| Time  ")
    print("----------------------------------------------------")
    
    
    for cl in cl_types:
        # ----------------------------------------------------------------------------------------------- #
        # ------------------------ Sort training gs by their score if curriculum ------------------------ #
        # ----------------------------------------------------------------------------------------------- #

        if cl in ["CL", "ACL", "PCL", "PACL"]:
            score_it = num_iters-1
            ascending = True if cl in ["CL","PCL"] else False
            gs_train, labels_train, j_train = sort_gs(weights_ncl[score_it], np.array(bias_ncl[score_it]), gs_train, labels_train, j_train, ascending, cl, nqubits, loss_type, gate_id, gen_basis)

        if cl == "RAND":
            scores = np.random.uniform(size=len(gs_train))
            p = scores.argsort()
            gs_train = gs_train[p]
            labels_train = labels_train[p]
            j_train = j_train[p]
            
        # ------------------------------------------------------------ #
        # ------------------------ Train QCNN ------------------------ #
        # ------------------------------------------------------------ #

        start_time = time.time()

        weights, \
        bias, \
        losses, \
        pred_train_arr, \
        pred_val_arr, \
        acc_train_arr, \
        acc_val_arr, \
        cv_j_train = train_qcnn(
            gs_train,
            gs_val,
            labels_train,
            labels_val,
            j_train,
            opt,
            cl,
            nqubits,
            num_iters,
            max_weight_init,
            nweights,
            initialization,
            with_bias,
            with_val,
            cl_pace,
            batch_size,
            optimizer,
            loss_type,
            gate_id,
            gen_basis,
        )

        run_time = time.time() - start_time
        
        if cl == "NCL":
            weights_ncl = weights
            bias_ncl = bias

        # --------------------------------------------------------- #
        # ------------------- Save calculations ------------------- #
        # --------------------------------------------------------- #
        save_data(time_now,
                folder_name,
                run,
                weights,
                bias,
                losses,
                cv_j_train,
                j_val,
                pred_train_arr,
                pred_val_arr,
                acc_train_arr,
                acc_val_arr,
                run_time,
                cl,
                with_val,
                )

    print("----------------------------------------------------")
    print()