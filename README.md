# Learning complexity gradually in quantum machine learning models
This repository contains the functions and scripts used for the simulation and optimization of quantum neural networks supporting the research presented in the paper "Learning complexity gradually in quantum machine learning models", available on [arxiv](https://arxiv.org/abs/2411.11954). Dependencies are listed in `requirements.txt`.

## Repository structure and running the code
This repository contains various utility files and three main executable scripts:

1. `main.py`: This is the core script where training and optimization occur. Before running, adjust the hyperparameters at the start of the script to suit your needs. When executed, this script generates multiple CSV files in the `Results` folder, containing relevant training data, and creates loss plots to visualize the progress of each training strategy.

2. `plot_accuracy.py`: This script generates a figure (saved in the `Results` folder) that compares the accuracies across different training strategies. Customize the parameters within the script to specify the target files, then run the script to produce the accuracy comparison figure.

3. `plot_probabilities.py`: This script outputs a figure illustrating the classification probabilities for different classes along a path in the parametrized Hamiltonian space. Adjust the settings as needed to obtain the desired visualization. The output is also saved in the `Results` folder.

In the Results folder, you’ll find examples of the outputs generated by these scripts. Note that, due to size constraints, the larger CSV files containing raw data have not been included in this repository.
