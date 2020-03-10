"""
This python script runs COMPAS using the Adaptive Importance Sampling (AIS) distributions
defined in phase 2, defined by STROOPWAFELgaussians.txt
It is based in part on the jupyter notebook ImportanceSampling-Step3-PostProcessing.ipynb written by Floor Broekgaarden
Modified by Simon Stevenson on 21st June 2018 to automate AIS
"""
import os
import pythonSubmit as ps 
import numpy as np
import sys

#-- For command line arguments
import argparse
parser = argparse.ArgumentParser(description="Run step 3 of AIS")
parser.add_argument("--masterFolderDir", help="Directory to masterFolder")
args = parser.parse_args()

"""
We need a copy of the *original* pythonSubmit and the STROOPWAFELgaussians.txt file
We change some settings in the pythonSubmit for the sampling phase and then run COMPAS as normal
"""

#-- Location of master folder
masterFolderDir = args.masterFolderDir

# Get a copy of STROOPWAFELgaussians.txt
bashCommand = 'cp ' + os.path.join(masterFolderDir, "STROOPWAFELgaussians.txt") + " ." 
print(bashCommand)
os.system(bashCommand)

# Get a copy of the *original* pythonSubmit
programOptions = ps.pythonProgramOptions()

# Get the hpc_input.py settings
sys.path.insert(0, "../../") 
from compas_hpc_input import *

# get how many samples are already run during exploratory phase
Nexpl = sum(1 for line in open('../../STROOPWAFELexploration/allSystems.dat'))-3 #  read in nr of lines, -3 for header

# Total nr of samples wanted
Ntot = programOptions.number_of_binaries * nBatches
# Total samples needed in refinement phase STROOPWAFEL
Nneeded = Ntot - Nexpl 
# Devide this over the batches and round up
programOptions.number_of_binaries = int(np.ceil(float(Nneeded) / nBatches)) 
#print programOptions.number_of_binaries, "number of binaries" 
#-- Turn off Exploratory phase
programOptions.AIS_exploratory_phase = False

#-- Turn on Refinement phase
programOptions.AIS_refinement_phase = True

#-- Run COMPAS using the modified pythonSubmit
ps.runCompas(programOptions)

