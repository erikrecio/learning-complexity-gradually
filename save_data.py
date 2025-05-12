import numpy as np
import jax.random
import jax.scipy.optimize
jax.config.update("jax_enable_x64", True)

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.ioff()

import os
import ast
import pandas as pd
from pypdf import PdfWriter
from ground_states import *

def save_multi_image(filename: str) -> None:
    """
    Saves all open matplotlib figures to a single PDF file.

    Parameters:
    filename (str): The path to the PDF file where figures will be saved.
    """
    pp = PdfPages(filename)  # Create a PdfPages object to save figures
    fig_nums = plt.get_fignums()  # Get a list of all open figure numbers
    figs = [plt.figure(n) for n in fig_nums]  # Retrieve figure objects by their numbers
    
    for fig in figs:
        fig.savefig(pp, format='pdf')  # Save each figure to the PDF
    
    pp.close()  # Close the PdfPages object

def close_all_figures() -> None:
    """
    Closes all open matplotlib figures.

    """
    fig_nums = plt.get_fignums()
    for n in fig_nums:
        fig = plt.figure(n)     # Get figure from number
        plt.close(fig)          # Close the figure object

        
def save_plots(
    time_now: str,
    folder_name: str,
    file_name: str,
    plot_run: int,
    acc_train: list[float],
    acc_val: list[float],
    losses: list[float],
    pred_train: list[np.ndarray],
    pred_val: list[np.ndarray],
    j_train: np.ndarray,
    j_val: np.ndarray,
    with_val: bool
) -> None:
    """
    Saves the accuracy, loss, and decision boundary plots to a PDF file.

    Parameters:
    time_now (str): The current time as a string.
    folder_name (str): The name of the folder where to save the plots.
    file_name (str): The name of the file where to save the plots.
    plot_run (int): The number of the current run.
    acc_train (list[float]): The accuracy of the training set at each iteration.
    acc_val (list[float]): The accuracy of the validation set at each iteration.
    losses (list[float]): The loss of the training set at each iteration.
    pred_train (list[np.ndarray]): The predictions of the training set at each iteration.
    pred_val (list[np.ndarray]): The predictions of the validation set at each iteration.
    j_train (np.ndarray): The coordinates of the training points.
    j_val (np.ndarray): The coordinates of the validation points.
    with_val (bool): Whether to include validation points in the plot or not.
    """
    fig, axis = plt.subplots(1,3)
    fig.set_figheight(6.5)
    fig.set_figwidth(20)
    fig.tight_layout(pad=2, w_pad=3.5)

    # ---------------------------------------------------------------------- #
    # -------------------- Loss and accuracy figure ------------------------ #
    # ---------------------------------------------------------------------- #

    iterations = range(1, len(acc_train)+1)

    color1 = 'darkred'
    axis[0].set_xlabel('Iterations')
    axis[0].set_ylabel('Accuracy %', color=color1)
    axis[0].plot(iterations, acc_train, label="Training", color=color1)
    axis[0].plot(iterations, acc_val, '-.', label="Validation", color=color1)
    axis[0].tick_params(axis='y', labelcolor=color1)
    axis[0].set_ylim(0,100)

    ax2 = axis[0].twinx()  # instantiate a second axes that shares the same x-axis

    color2 = 'darkblue'
    ax2.set_ylabel('Loss', color=color2)  # we already handled the x-label with axis[0]
    ax2.plot(iterations, losses, label="Loss", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    # ax2.set_ylim(bottom=0)

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.legend()
    axis[0].set_title(f"Accuracy and Loss - Run {plot_run}")


    # ----------------------------------------------------------------------------- #
    # ---------------------------- Training points -------------------------------- #
    # ----------------------------------------------------------------------------- #

    # define regions coordinates
    x01, y01 = region01_coords[:,0], region01_coords[:,1]
    x02, y02 = region02_coords[:,0], region02_coords[:,1]
    x1, y1 = region1_coords[:,0], region1_coords[:,1]
    x2, y2 = region2_coords[:,0], region2_coords[:,1]
    x3, y3 = region3_coords[:,0], region3_coords[:,1]

    # put the regions into the plot
    axis[1].fill(x01, y01, facecolor='lightskyblue')    # class 0
    axis[1].fill(x02, y02, facecolor='lightskyblue')    # class 0
    axis[1].fill(x1, y1, facecolor='sandybrown')        # class 1
    axis[1].fill(x2, y2, facecolor='salmon')            # class 2
    axis[1].fill(x3, y3, facecolor='lightgreen')        # class 3

    pred_train_plot = np.array(pred_train[-1])
    pred_val_plot = np.array(pred_val[-1])

    colors = ["b", "orange", "r", "g"]

    # plot datapoints
    for i in range(4):
        axis[1].scatter(
            j_train[:, 0][pred_train_plot==i],
            j_train[:, 1][pred_train_plot==i],
            c=colors[i],
            marker="o",
            edgecolors="k",
            label=f"class {i+1} train",
        )
        # if with_val:
        #     axis[1].scatter(
        #         j_val[:, 0][pred_val_plot==i],
        #         j_val[:, 1][pred_val_plot==i],
        #         c=colors[i],
        #         marker="^",
        #         edgecolors="k",
        #         label=f"class {i+1} validation",
        #     )


    # plt.legend()
    axis[1].set_title(f"Training ({acc_train[-1]:.0f}%)")


    # ------------------------------------------------------------------------------ #
    # ---------------------------- Validation points ------------------------------- #
    # ------------------------------------------------------------------------------ #

    # define regions coordinates
    x01, y01 = region01_coords[:,0], region01_coords[:,1]
    x02, y02 = region02_coords[:,0], region02_coords[:,1]
    x1, y1 = region1_coords[:,0], region1_coords[:,1]
    x2, y2 = region2_coords[:,0], region2_coords[:,1]
    x3, y3 = region3_coords[:,0], region3_coords[:,1]

    # put the regions into the plot
    axis[2].fill(x01, y01, facecolor='lightskyblue')    # class 0
    axis[2].fill(x02, y02, facecolor='lightskyblue')    # class 0
    axis[2].fill(x1, y1, facecolor='sandybrown')        # class 1
    axis[2].fill(x2, y2, facecolor='salmon')            # class 2
    axis[2].fill(x3, y3, facecolor='lightgreen')        # class 3

    pred_train_plot = np.array(pred_train[-1])
    pred_val_plot = np.array(pred_val[-1])

    colors = ["b", "orange", "r", "g"]

    # plot datapoints
    for i in range(4):
        # axis[2].scatter(
        #     j_train[:, 0][pred_train_plot==i],
        #     j_train[:, 1][pred_train_plot==i],
        #     c=colors[i],
        #     marker="o",
        #     edgecolors="k",
        #     label=f"class {i+1} train",
        # )
        if with_val:
            axis[2].scatter(
                j_val[:, 0][pred_val_plot==i],
                j_val[:, 1][pred_val_plot==i],
                c=colors[i],
                marker="^",
                edgecolors="k",
                label=f"class {i+1} validation",
            )


    # plt.legend()
    axis[2].set_title(f"Validation ({acc_val[-1]:.0f}%)")

    # ---------------------------------------------------------------------- #
    # --------------------------- Save plots ------------------------------- #
    # ---------------------------------------------------------------------- #

    plots_pdf_name = f"{folder_name}/{time_now} - Losses - {file_name}.pdf"
    
    
    # If the file doesn't exist we save it. If it does, we merge it.
    if not os.path.isfile(plots_pdf_name):
        save_multi_image(plots_pdf_name)
    
    else:
        save_multi_image(plots_pdf_name + "2")
        # Merge the new plot with the rest and delete the last file
        merger = PdfWriter()
        merger.append(plots_pdf_name)
        merger.append(plots_pdf_name + "2")
        merger.write(plots_pdf_name)
        merger.close()
        os.remove(plots_pdf_name + "2")
    
    close_all_figures()
    
    
def save_hyperparameters(
    time_now: str,
    folder_name: str,
    file_name: str,
    num_iters: int,
    num_runs: int,
    cl_types: list,
    with_val: bool,
    nqubits: int,
    with_bias: bool,
    optimizer: str,
    loss_type: str,
    initialization: str,
    ham: str,
    batch_size: int,
    train_size: int,
    val_size: int,
    cl_pace_ratios: list[float],
    cl_iter_ratios: list[float],
    gen_type: str,
    uniform_train: bool,
    uniform_val: bool,
    epsilon_train: bool,
    epsilon_val: bool,
    max_weight_init: float,
    stepsize: float,
    gate_id: str,
) -> None:
    """
    Saves the hyperparameters of a training session to a CSV file.

    Parameters:
    time_now (str): The current timestamp as a string.
    folder_name (str): The name of the folder where the CSV file will be saved.
    file_name (str): The base name of the file to save the hyperparameters.
    num_iters (int): Number of iterations.
    num_runs (int): Number of runs.
    cl_types (list): List of curriculum learning types.
    with_val (bool): Whether validation is included.
    nqubits (int): Number of qubits.
    with_bias (bool): Whether to include bias.
    optimizer (str): Optimizer name.
    loss_type (str): Type of loss function.
    initialization (str): Initialization method.
    ham (str): Hamiltonian type.
    batch_size (int): Size of the training batch.
    train_size (int): Size of the training set.
    val_size (int): Size of the validation set.
    cl_pace_ratios (list[float]): Pace ratios for curriculum learning.
    cl_iter_ratios (list[float]): Iteration ratios for curriculum learning.
    gen_type (str): Generator type.
    uniform_train (bool): Use uniform distribution for training.
    uniform_val (bool): Use uniform distribution for validation.
    epsilon_train (bool): Use epsilon-deflated regions for training.
    epsilon_val (bool): Use epsilon-deflated regions for validation.
    max_weight_init (float): Maximum weight initialization.
    stepsize (float): Step size for optimization.

    Returns:
    None
    """
    
    # Dictionary to store hyperparameters
    hyperparameters = {
        "num_iters": [num_iters],
        "num_runs": [num_runs],
        "cl_types": [cl_types],
        "with_val": [with_val],
        "nqubits": [nqubits],
        "with_bias": [with_bias],
        "optimizer": [optimizer],
        "loss_type": [loss_type],
        "initialization": [initialization],
        "hamiltonian": [ham],
        "batch_size": [batch_size],
        "train_size": [train_size],
        "val_size": [val_size],
        "cl_pace_ratios": [cl_pace_ratios],
        "cl_iter_ratios": [cl_iter_ratios],
        "generator_type": [gen_type],
        "uniform_train": [uniform_train],
        "uniform_val": [uniform_val],
        "epsilon_train": [epsilon_train],
        "epsilon_val": [epsilon_val],
        "max_weight_init": [max_weight_init],
        "stepsize": [stepsize],
        "key": [time_now],
        "gate_id": [gate_id],
    }

    # Convert dictionary to DataFrame
    hyperparameters_df = pd.DataFrame(hyperparameters)

    # Construct the CSV file path
    hyperparameters_file_name = f"{folder_name}/{time_now} - Hyperparameters{file_name}.csv"

    # Save the DataFrame to a CSV file
    hyperparameters_df.to_csv(hyperparameters_file_name, index=False)
    
    
def save_data(
    time_now: str,
    folder_name: str,
    run: int,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    losses: list[float],
    j_train: jnp.ndarray,
    j_val: jnp.ndarray,
    pred_train: list[np.ndarray],
    pred_val: list[np.ndarray],
    acc_train: list[float],
    acc_val: list[float],
    run_time: float,
    cl: str,
    with_val: bool,
) -> None:
    """
    Saves the data of the run to a CSV file.

    Parameters:
    time_now (str): The current time as a string.
    folder_name (str): The name of the folder where to save the data.
    run (int): The number of the current run.
    weights (jnp.ndarray): The weights of the variational quantum circuit.
    bias (jnp.ndarray): The bias of the variational quantum circuit.
    losses (list[float]): The loss of the training set at each iteration.
    j_train (jnp.ndarray): The coordinates of the training points.
    j_val (jnp.ndarray): The coordinates of the validation points.
    pred_train (list[np.ndarray]): The predictions of the training set at each iteration.
    pred_val (list[np.ndarray]): The predictions of the validation set at each iteration.
    acc_train (list[float]): The accuracy of the training set at each iteration.
    acc_val (list[float]): The accuracy of the validation set at each iteration.
    run_time (float): The time it took to run the current iteration.
    cl (str): The type of curriculum learning used.
    with_val (bool): Whether validation is included.
    """
    
    # -------------------- Total Data -------------------- #
    it_max = np.argmax(np.array(acc_train))
    
    data = {
        "run": run,
        "it_max": it_max,
        "acc_train_max": acc_train[it_max],
        "acc_train_last": acc_train[-1],
        "acc_val_max": acc_val[it_max],
        "acc_val_last": acc_val[-1],
        "run_time": run_time,
        "weights": [weights],
        "bias": [bias],
        "losses": [losses],
        "j_train": [j_train.tolist()],
        "j_val": [j_val.tolist()],
        "pred_train": [pred_train],
        "pred_val": [pred_val],
        "acc_train": [acc_train],
        "acc_val": [acc_val],
    }

    data = pd.DataFrame(data)
    
    data_file_name = f"{folder_name}/{time_now} - Data - {cl}.csv"
    data.to_csv(data_file_name, index=False, mode='a', header = not os.path.exists(data_file_name))
    
    
    # ------------------- Results ------------------- #
    
    read_data = pd.read_csv(
        data_file_name,
        usecols=[
            "it_max",
            "acc_train_max",
            "acc_val_max",
            "acc_train",
            "acc_val"
        ],
        converters={
            "acc_train": ast.literal_eval,
            "acc_val": ast.literal_eval
        }
    )
    
    total_it_max = read_data["it_max"]
    total_acc_train_max = read_data["acc_train_max"]
    total_acc_val_max = read_data["acc_val_max"]
    total_acc_train = read_data["acc_train"].tolist()
    total_acc_val = read_data["acc_val"].tolist()
    
    best_run_max = total_acc_train_max.argmax()
    best_it_max = total_it_max[best_run_max]
    avg_acc_train_max = total_acc_train_max.mean()
    avg_acc_val_max = total_acc_val_max.mean()
    
    best_run_last = np.argmax(np.array(total_acc_train)[:, -1])
    avg_acc_train_last = np.mean(np.array(total_acc_train)[:, -1])
    avg_acc_val_last = np.mean(np.array(total_acc_val)[:, -1])

    results = {
        "type_cl": [cl],
        "num_runs": [run + 1],
        "best_run_max": [best_run_max],
        "best_run_last": [best_run_last],
        "best_it_max": [best_it_max],
        "best_it_last": [-1],
        "best_acc_train_max": [total_acc_train[best_run_max][best_it_max]],
        "best_acc_train_last": [total_acc_train[best_run_last][-1]],
        "best_acc_val_max": [total_acc_val[best_run_max][best_it_max]],
        "best_acc_val_last": [total_acc_val[best_run_last][-1]],
        "avg_acc_train_max": [avg_acc_train_max],
        "avg_acc_train_last": [avg_acc_train_last],
        "avg_acc_val_max": [avg_acc_val_max],
        "avg_acc_val_last": [avg_acc_val_last]
    }
    results = pd.DataFrame(results)
    results_file_name = f"{folder_name}/{time_now} - Results.csv"
    
    # If file exists, we update the info
    if os.path.exists(results_file_name):
        read_results = pd.read_csv(results_file_name)
        row_index = read_results.loc[read_results["type_cl"] == cl].index
        
        if row_index.shape != (0,):
            read_results.drop(labels=row_index[0], axis=0, inplace=True) # we delete the line if it already exists
            
        results = pd.concat([read_results, results], ignore_index=True)
    
    results.to_csv(results_file_name, index=False)
    
    
    
    # ------------------- Plots ------------------- #
    # save_plots(time_now,
    #            folder_name,
    #            cl,
    #            run,
    #            acc_train,
    #            acc_val,
    #            losses,
    #            pred_train,
    #            pred_val,
    #            j_train,
    #            j_val,
    #            with_val,
    #           )
    
    if cl == "NCL":
        cl_str = "NCL   "
    elif cl=="CL":
        cl_str = "CL    "
    elif cl=="ACL":
        cl_str = "ACL   "
    elif cl=="SPCL":
        cl_str = "SPCL  "
    elif cl=="SPACL":
        cl_str = "SPACL "
    elif cl=="PCL":
        cl_str = "PCL   "
    elif cl=="PACL":
        cl_str = "PACL  "
    elif cl=="FSPCL":
        cl_str = "FSPCL "
    elif cl=="FSPACL":
        cl_str = "FSPACL"
    elif cl=="RAND":
        cl_str = "RAND  "
        
    print(
        f" {cl_str} |"
        f" {run:3d} |"
        f" {it_max:4d}/{len(acc_train)-1:4d} |"
        f"  {acc_train[it_max]:0.0f}/{acc_train[-1]:0.0f}  |"
        f" {acc_val[it_max]:0.0f}/{acc_val[-1]:0.0f} |"
        f" {run_time:0.0f}"
    )

    
    
