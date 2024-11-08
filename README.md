# Learning complexity gradually in quantum machine learning models
In this repository you will find all the functions and scripts used for the simulation of Parametrized Quantum Circuits and their optimization with Curriculum Learning needed to write the paper "Learning complexity gradually in quantum machine learning models" that you can find on [arxiv](https://www.google.com). All the packages required can be found in requirements.txt.

## Repository structure and running the code
This repository has some files filled with functions, but only three executable scripts: 

1. `main.py`: The main part of the code where the training optimization happens. At the start of the script, set all the hyperparameter settings as you want and then run the script. This will create several csv with relevant data from the training in the `Results` folder. It will also create some loss plots in order to better visualize the progress of each strategy.

2. `plot_accuracy.py`: A small script that outputs a figure (in the Results folder) comparing the accuracies of several Curriculum Learning strategies. In the settings, choose the parameters that point to the right files and run the script.

3. `plot_probabilities.py`: Another small script that outputs, this time, a figure showing the probabilities of classification of the different classes along a certain path in the parametrized Hamiltonian space.

You can see already an example of the output of these three scripts in the Results folder. Due to its large size, the Data csv are spared from being uploaded into github and are therefore missing.
