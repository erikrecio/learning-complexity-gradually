import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jax
import ast

from ground_states import ground_state
from loss import variational_classifier

# ---------------------- START SETTINGS ---------------------- #

nqubits = 8
num_iters = 600
time_now = "2024-11-08 10-10-40" # Put the date and time of your calculations here
file_name = "FSPACL"

run = 0
iteration = -1
num_points = 100

# ---------------------- END SETTINGS ---------------------- #

folder_name = f"Results/{nqubits}q - {num_iters:} iters"
hyper_file_name = f"{folder_name}/{time_now} - Hyperparameters.csv"
data_file_name = f"{folder_name}/{time_now} - Data - {file_name}.csv"

# Read hyperparameters
hyper = pd.read_csv(hyper_file_name)
ham = hyper["hamiltonian"][0]
gate_id = hyper["gate_id"][0]

# Read the saved data
read_data = pd.read_csv(data_file_name,
                        usecols=["weights", "bias"],
                        converters={"weights":ast.literal_eval,
                                    "bias":ast.literal_eval})
weights = read_data["weights"][run][iteration]
bias = read_data["bias"][run][iteration]

if ham == "gch":
    xx = 1*np.ones(num_points)
    yy = np.linspace(-4, 4, num_points)
elif ham == "ssh":
    xx = np.linspace(0, 3, num_points) 
    yy = 3*np.ones(num_points)

X_grid = np.array([np.array([x, y]) for x, y in zip(xx, yy)])

gs = jax.vmap(ground_state, in_axes=(0,0,None, None))(X_grid[:,0], X_grid[:,1], nqubits, ham)
probs = np.array([variational_classifier(weights, np.array(bias), g, nqubits, gate_id) for g in gs])

fig, axis = plt.subplots(1,1)
fig.set_figheight(5)
fig.set_figwidth(6)
fig.tight_layout()

x_axis = yy

p00 = probs[:,0]
p01 = probs[:,1]
p10 = probs[:,2]
p11 = probs[:,3]

if ham == "gch":
    axis.plot(x_axis, p00, label="prob class 0", color="blue")
    axis.plot(x_axis, p01, label="prob class 1", color="orange")
    axis.plot(x_axis, p10, label="prob class 2", color="red")
    axis.plot(x_axis, p11, label="prob class 3", color="green")
    axis.set_xlabel(r'$j_2$')
    axis.set_title(file_name + r", $j_1 = 1$")
elif ham == "ssh":
    axis.plot(x_axis, p00, label="prob class null", color="grey")
    axis.plot(x_axis, p01, label="prob class 1", color="red")
    axis.plot(x_axis, p10, label="prob class 2", color="blue")
    axis.plot(x_axis, p11, label="prob class 3", color="orange")
    axis.set_xlabel(r'$j_1$/$j_2$')
    axis.set_title(file_name + r", $\delta = 3$")

axis.set_ylabel('probability')
plt.legend(loc="lower right")
plt.savefig(f"{folder_name}/{time_now} - Probabilities - {file_name}.pdf", bbox_inches='tight')
