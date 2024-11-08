from matplotlib import pyplot as plt
import pandas as pd
import ast
import numpy as np
import scienceplots
import latex

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------- START SETTINGS ---------------------- #

nqubits = 8
num_iters = 600
time_now = "2024-11-08 11-26-02" # Put the date and time of your calculations here

# In files_names, put all the different CL strategies you want to compare. Examples:
file_names = ["NCL","RAND","CL","ACL"] #1
# file_names = ["NCL","SPCL","SPACL","FSPACL"] #2
# file_names = ["NCL","RAND"] #3

# ---------------------- END SETTINGS ---------------------- #

folder_name = f"Results/{nqubits}q - {num_iters:} iters"
data_file_names = []
for file_name in file_names:
    data_file_names.append(f"{folder_name}/{time_now} - Data - {file_name}.csv")

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fontfig = 9
plt.rcParams.update({'font.size': fontfig})
plt.rcParams.update({'font.family': 'times'})


with plt.style.context(['science', 'std-colors']):
    
    plt.rcParams['axes.linewidth'] = 1.10
    fig, axis = plt.subplots(1,1)
    fig.set_figheight(5)
    fig.set_figwidth(6)
    fig.tight_layout()
    
    i=0
    for data_file_name in data_file_names:
        
        # Read the saved data #####################
        read_data = pd.read_csv(data_file_name,
                                usecols=["losses",
                                        "acc_train",
                                        "acc_val"],
                                converters={"losses":ast.literal_eval,
                                        "acc_train":ast.literal_eval,
                                        "acc_val":ast.literal_eval})

        all_runs_losses = np.array(list(map(np.array, read_data["losses"])))
        all_runs_acc_train = np.array(list(map(np.array, read_data["acc_train"])))
        all_runs_acc_val = np.array(list(map(np.array, read_data["acc_val"])))
        
        # We take the averages
        losses = np.mean(all_runs_losses, axis=0)
        acc_train = np.mean(all_runs_acc_train, axis=0)
        acc_val = np.mean(all_runs_acc_val, axis=0)

        std_losses = np.std(all_runs_losses, axis=0)
        std_acc_train = np.std(all_runs_acc_train, axis=0)
        std_acc_val = np.std(all_runs_acc_val, axis=0)
        
        iterations = range(1, len(read_data["losses"][0])+1)

        axis.plot(iterations, acc_train, label=file_names[i]+" - Training", color=colors[i])
        axis.fill_between(iterations, acc_train - std_acc_train/2, acc_train + std_acc_train/2, color=colors[i], alpha=0.15)
        axis.plot(iterations, acc_val, '-.', label=file_names[i]+" - Validation", color=colors[i])
        axis.fill_between(iterations, acc_val - std_acc_val/2, acc_val + std_acc_val/2, color=colors[i], alpha=0.15)
        i+=1
    
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Accuracy %')
    axis.set_ylim(0,100)
    axis.set_title(f"Average over {len(all_runs_acc_val)} runs with {nqubits} qubits")                
    plt.legend(loc="lower right")
    plt.savefig(f"{folder_name}/{time_now} - Accuracy - {file_names}.pdf", bbox_inches='tight')