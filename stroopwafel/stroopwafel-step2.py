''' 
Step 2 of Adaptive Importance Sampling
Author: Floor Broekgaarden | fsbroekgaarden@gmail.com
date: 03-Oct-2018 
Modified by Simon Stevenson 21st June 2018 to automate AIS

####    SUMMARY    ####
#   This file uses the results of the exploratory phase to define the gaussians for the instrumental distribution (Eq. 8 in Broekgaarden+18)
#   The result of running this python file will be a textfile AISsamples.txt in the masterFolder 
#   This file contains the means and std (sigma) for the Gaussian distributions o the instrumental distribution that COMPAS uses to sample from in the remaingin simulation.

#   Note that the covariances in AISgaussians.txt  here are similar to the definition of gsl_ran_norml() of C++ 
#   see https://www.gnu.org/software/gsl/manual/html_node/The-Gaussian-Distribution.html (so not sigma^2(x) as in python normal distributions) 
'''

from __future__ import division
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy import integrate
import math
import random
import os
import pythonSubmit as ps


########################################################################
# Read in the properties of the parameter space and target distribution
########################################################################

# add directory of the allDoubleCompactObjects.dat and allSystems.dat files:
# os.path.dirname(os.path.dirname(os.getcwd())) gets the folder above the current working directory
outputDirectory = os.path.join(os.path.dirname(os.getcwd()), "STROOPWAFELexploration")
masterDirectory = os.getcwd() # masterFolder

## Get a copy of the program options
programOptions = ps.pythonProgramOptions()

## For convenience define:
initial_mass_function = programOptions.initial_mass_function
initial_mass_min =  programOptions.initial_mass_min
initial_mass_max =  programOptions.initial_mass_max

initial_mass_power = programOptions.initial_mass_power

semi_major_axis_distribution = programOptions.semi_major_axis_distribution
semi_major_axis_min =  programOptions.semi_major_axis_min  
semi_major_axis_max = programOptions.semi_major_axis_max

mass_ratio_distribution = programOptions.mass_ratio_distribution
mass_ratio_min = programOptions.mass_ratio_min
mass_ratio_max = programOptions.mass_ratio_max


# Choice on DCO types included
DCOtype = programOptions.AIS_DCOtype #'BHNS'  # 'all', 'BNS', 'BBH'

# Choice on whether to exclude binaries which
# merge in over a Hubble time
Hubble = programOptions.AIS_Hubble  #True # False

# Choice on whether to exclude binaries where the secondary
# has undergone RLOF after a CEE True = exclude RLOFSecondaryAfterCEE
RLOF = programOptions.AIS_RLOF #True  # False

# Choice on whether to exclude binaries which use the optimistic
# CE prescription: if Pessimistic = False equals Optimistic
Pessimistic = programOptions.AIS_Pessimistic #False  # True

# define factor to multiply width of Gaussins with
factor = programOptions.kappa_gaussians



dimension = 3  # nr of parameters used [Now fixed]


#################################################
#################################################

# define slope IMF -- now assumes KROUPA IMF. But its not really a problem if you use another distribution. 
alpha_IMF = 2.3    
K1 = (-alpha_IMF +1) / (initial_mass_max**(-alpha_IMF+1) - initial_mass_min**(-alpha_IMF+1)) # Normalization constant of the Kroupa IMF 

# Functions to approximate average distance between two samples in initial parameter space
# Needed for M1 since inverse CDF does not exist for the Kroupa IMF
def InverseM1(M1array):
    '''Project the initial mass samples on a unit line [0,1] '''
    InverseM1 = (((K1/M1array)**(1/alpha_IMF) )  - ((K1/initial_mass_min)**(1/alpha_IMF) ) ) / (((K1/initial_mass_max)**(1/alpha_IMF) ) - ((K1/initial_mass_min)**(1/alpha_IMF) ))
    return InverseM1

def BackToM1(MinverseArray):
    '''Transform the distance between the samples on [0,1] back to M_1 space '''
    M1array = K1 / (((MinverseArray * (((K1/initial_mass_max)**(1/alpha_IMF) ) - ((K1/initial_mass_min)**(1/alpha_IMF) )) ) + ((K1/initial_mass_min)**(1/alpha_IMF) ) )**(alpha_IMF)) 
    return M1array
    


print('-------------------------------------------------')
print('  Started creating the gaussians \n')

###########################
#  In the next block of code a mask is created that filters all DCO from the exploratory phase with certain properties.
##########################

# data needed for mask #'/systemParameters.txt', /allDoubleCompactObjects.txt
d_all = np.genfromtxt(os.path.join(outputDirectory, "allSystems.dat"), names=True, skip_header=2)
d_merg = np.genfromtxt(os.path.join(outputDirectory, "allDoubleCompactObjects.dat"), names=True, skip_header=2)

# DCO mask
if DCOtype == 'BNS' or  DCOtype == 'DNS' or  DCOtype == 'NSNS':
    mask0 = np.logical_and(d_merg["stellarType1"] == 13,  d_merg["stellarType2"] == 13)

if DCOtype == 'BHNS' or DCOtype == 'NSBH':
    mask0 = (np.logical_and(d_merg['stellarType1'] == 14, d_merg['stellarType2'] == 13) +
                    np.logical_and(d_merg['stellarType1'] == 13, d_merg['stellarType2'] == 14))

if DCOtype == 'BBH' or DCOtype == 'BHBH':
    mask0 = np.logical_and(d_merg["stellarType1"] == 14,  d_merg["stellarType2"] == 14)

if DCOtype == 'ALL' or DCOtype == 'all':
    mask0 = (np.logical_and(d_merg["stellarType1"] == 14,  d_merg["stellarType2"] == 14) +
                     np.logical_and(d_merg['stellarType1'] == 14, d_merg['stellarType2'] == 13) +
                     np.logical_and(d_merg['stellarType1'] == 13, d_merg['stellarType2'] == 14) +
                     np.logical_and(d_merg['stellarType1'] == 13, d_merg['stellarType2'] == 13))
# Hubble mask
if Hubble:
     mask1 = (d_merg["mergesInHubbleTimeFlag"] == 1)
elif not Hubble:
    mask1 = (d_merg["mergesInHubbleTimeFlag"] == 1) + (d_merg["mergesInHubbleTimeFlag"] == 0)
# RLOF mask
if RLOF:
     mask2 = np.logical_not(d_merg["RLOFSecondaryAfterCEE"] == 1)
elif not RLOF:
    mask2 = np.logical_not(d_merg["RLOFSecondaryAfterCEE"] == 1) + \
    np.logical_not(d_merg["RLOFSecondaryAfterCEE"] == 0)
# Pessimistic / Optimistic mask 
if Pessimistic:
     mask3 = np.logical_not(d_merg["optimisticCEFlag"] == 1)
elif not Pessimistic:
    mask3 = np.logical_not(d_merg["optimisticCEFlag"] == 1) + \
    np.logical_not(d_merg["optimisticCEFlag"] == 0) 
# Define mask that combines the 4 mask options
combinedMask = np.logical_and(np.logical_and(mask0 ==1, mask1 ==1) , np.logical_and(mask2 ==1, mask3 ==1))


# get the initial parameters of the DCOs of interest
M1ZAMS = d_merg["M1ZAMS"][combinedMask]
q = d_merg["M2ZAMS"][combinedMask] / M1ZAMS
# we are sampling and drawing Gaussians in Log10 space
separationInitial = np.log10(d_merg["separationInitial"][combinedMask])

# get the initial parameters of all samples in the exploratory phase
M1ZAMSall = d_all["mass1"]
qall = d_all["mass2"] / M1ZAMSall
# we are sampling and drawing Gaussians in Log10 space
separationInitialall = np.log10(d_all["separation"])
Nini = len(M1ZAMSall)

###########################
'''Means of the Gaussians'''
# the means of the gaussians are the locations of the DCO of interest from the exploratory phase
###########################

# add the nr of gaussians as first element since this is used in COMPAS to initialize the vectors of Gaussians faster   
# for all second elements just add a [0] in the beginning, we will not read this in. 
MEAN_M1cpp = np.concatenate(([int(len(M1ZAMS))], M1ZAMS))  
MEAN_acpp = np.concatenate(([int(0)], separationInitial)) # 
MEAN_qcpp = np.concatenate(([int(0)], q))


###########################
'''Covariances of the Gaussians'''
# The covariance matrix is diagonal (see Eq. 9 in Broekgaarden+18)
# The covariances are  proportional to the average distance to the next sample
###########################

# Calculate average nr of samples in each sub dimension (Eq. 10 in Broekgaarden+18)
Daverage = 1 / (Nini)**(1/ dimension)

# Calculate average distance to next hit for M1 (by using above average difference between samples ini in unit cube)
InverseMuM1 = InverseM1(M1ZAMS) # invert to unit box (or line in 1D)
averageDrightM1 = np.abs(BackToM1(InverseMuM1+Daverage) - BackToM1(InverseMuM1)) 
averageDleftM1 = np.abs(BackToM1(InverseMuM1-Daverage) - BackToM1(InverseMuM1))
cov_M1 = np.maximum(averageDleftM1, averageDrightM1)   # take the max average distance to a point in the left or right 

# calculate average distance for q
cov_q = np.ones(len(M1ZAMS))*(Daverage) # Since q already uniform on [0,1]
# and average distance  for a 
cov_a = np.ones(len(M1ZAMS))*(Daverage * (np.log10(semi_major_axis_max) - np.log10(semi_major_axis_min)))

# add len as first element since this we will read in when in COMPAS (to initialize the vectors of Gaussians faster)   
# for all second elements just add a [0] in the beginning, we will not read this in. 
COV_M1cpp = factor * np.asarray(cov_M1)
COV_acpp =  factor * np.asarray(cov_a)
COV_qcpp =  factor * np.asarray(cov_q)

COV_M1cpp = np.concatenate(([int(0)], COV_M1cpp))
COV_acpp = np.concatenate(([int(0)], COV_acpp))
COV_qcpp = np.concatenate(([int(0)], COV_qcpp))

# save the gaussian distribution parameters in the masterFolder
np.savetxt(os.path.join(masterDirectory, "STROOPWAFELgaussians.txt") , np.c_[MEAN_M1cpp, MEAN_acpp, MEAN_qcpp, COV_M1cpp, COV_acpp, COV_qcpp],  header='  Msol    log10(AU)    #    Msol    log10(AU)    #    float    float    float    float    float    float    AISmeans_M1     AISmeans_a    AISmeans_q    std_M1    std_a    std_q' )

print(' \n The file STROOPWAFELgaussians.txt has been created. Step 2 of AIS for COMPAS is completed!')
print('-------------------------------------------------')
