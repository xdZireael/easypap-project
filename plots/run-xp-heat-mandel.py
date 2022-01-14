#!/usr/bin/env python3
from expTools import *

easypapOptions = {
    "-k ": ["mandel"],
    "-i ": [10],
    "-v ": ["omp_tiled"],
    "-s ": [1024,2048],
}

# OMP Internal Control Variable
ompICV = {
    "OMP_SCHEDULE=": ["dynamic","static,1"],
    "OMP_NUM_THREADS=": list(range(12, 0, -1)),
    "OMP_PLACES=":"cores"
}
nbrun = 1

easypapOptions["-th "] = [1]
easypapOptions["-tw "] = [256]
# Lancement des experiences
#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

easypapOptions["-tw "] = [128]
easypapOptions["-s "] = [512]
# Lancement des experiences
#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

ompICV["OMP_SCHEDULE="] = ["static,1"]
easypapOptions["-th "] = [2]
#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")


easypapOptions["-tw "] = [8]
easypapOptions["-th "] = [1]

easypapOptions["-s "] = [2048]
# Lancement des experiences
#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")



#del easypapOptions["-ts "]
#easypapOptions["--label "] = ["line"]
#easypapOptions["-th "] = [1]
#easypapOptions["-tw "] = [64, 512]

easypapOptions["--label "] = ["heat"]
easypapOptions["-th "] = [2 **i for i in range(0, 6)]
easypapOptions["-tw "] = [2 ** i for i in range(3,10)]
ompICV["OMP_SCHEDULE="] = ["dynamic","static,1"]
ompICV["OMP_NUM_THREADS="] = ["48"]

execute('./run ', ompICV, easypapOptions, nbrun, verbose=True, easyPath=".")


ompICV = {
    "OMP_SCHEDULE=": ["static,1"],
    "OMP_NUM_THREADS=": [1]
}




#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")

 #Lancement de la version seq avec le nombre de thread impose a 1

nbrun = 5
easypapOptions = {
    "-k ": ["mandel"],
    "-i ": [10],
    "-v ": ["seq"],
    "-s ": [2048],
     "--arg ": ["random"],
}
ompICV = {"OMP_NUM_THREADS=": [1]}
#execute('./run ', ompICV, easypapOptions, nbrun, verbose=False, easyPath=".")
