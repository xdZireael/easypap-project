#!/usr/bin/env python3
from expTools import *

# Recommended Plot:
# plots/easyplot.py -if mandel.csv -v omp_tiled -- col=schedule row=label
easypapOptions = {
    "-k": ["life"],
    "-i": [10],
    "-v": ["mpi_omp_border"],
    "-a": ["moultdiehard1398"],
    "-s": [4096],
    "-of": ["mpi_threads.csv"],
    "--label" : ["nb_proc"],
    "-mpi": ['"-np 3"']
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE": ["dynamic"],
    "OMP_PLACES": ["threads"],
    "OMP_NUM_THREADS": [1,4,8,12,16,20,25,30,35,40,44,48]
}

nbruns = 1
# Lancement des experiences
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")

ompICV = {
    "OMP_NUM_THREADS": [1]
}

easypapOptions["-v"] = ["seq"]
execute("./run ", ompICV, easypapOptions, nbruns, verbose=True, easyPath=".")


print("Recommended plot:")
print(" plots/easyplot.py -if mpi_threads.csv -v omp_tiled -- col=schedule row=label")
