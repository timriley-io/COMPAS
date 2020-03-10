"""
# Python script for post processing the weights of the stroopwafel sampled data. 
This is the script that can be implemented in the postProcessing.py of COMPAS.
**This jupyter notebook is part 3 of implementing adaptive importance sampling in COMPAS ** <br>
See Broekgaarden+18 and the AdaptiveImportanceSampling folder on gitlab. 
Last updated on 08-09-2018; Floor Broekgaarden with help of Simon Stevenson
***
"""
from __future__ import division
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy import integrate
import math
import random
import os
import pythonSubmit as ps
import h5py as h5 


########################################################################
# Read in the properties of the parameter space and target distribution
########################################################################

# add directory of the allDoubleCompactObjects.dat and allSystems.dat files:
# os.path.dirname(os.path.dirname(os.getcwd())) gets the folder above the current working directory
outputDirectoryExpl = os.path.join(os.path.dirname(os.getcwd()), "STROOPWAFELexploration")
outputDirectoryRef = os.path.join(os.path.dirname(os.getcwd()), "STROOPWAFELrefinement")
masterDirectory = os.getcwd() # masterFolder

## Get a copy of the program options
programOptions = ps.pythonProgramOptions()

## For convenience define:
initial_mass_function = programOptions.initial_mass_function
initial_mass_min =  programOptions.initial_mass_min
initial_mass_max =  programOptions.initial_mass_max

initial_mass_power = programOptions.initial_mass_power

semi_major_axis_distribution = programOptions.semi_major_axis_distribution
semi_major_axis_min =  np.log10(programOptions.semi_major_axis_min)
semi_major_axis_max = np.log10(programOptions.semi_major_axis_max)

mass_ratio_distribution = programOptions.mass_ratio_distribution
mass_ratio_min = programOptions.mass_ratio_min
mass_ratio_max = programOptions.mass_ratio_max

minimum_secondary_mass = programOptions.minimum_secondary_mass


#eccentricity, needed for RLOFonZAMS
eccentricity_distribution = programOptions.eccentricity_distribution
eMax, eMin = programOptions.eccentricity_min, programOptions.eccentricity_max


dimension = 3  # nr of parameters used [Now fixed]

if (initial_mass_function == 'KROUPA') & (initial_mass_min >= 0.5) & \
(mass_ratio_distribution == 'FLAT') & (semi_major_axis_distribution == 'FLATINLOG'):
    #slope of IMF
    alpha_IMF = 2.3    
    # print('Great! - your initial settigs are implemented in the post-processing for AIS')

else:
    # lower mass power law of Kroupa+01 is not implemented yet
    # other priors are not implemented yet
    print('Error: AIS post processing needs implementation of greater variety of priors - this is easy so just poke Floor :-) ')


dimension = 3  # nr of parameters used [Now fixed]



#pdfs initial parameters: 
K1 = (-alpha_IMF +1) / (initial_mass_max**(-alpha_IMF+1) - initial_mass_min**(-alpha_IMF+1))
def prior_M1(x):
    """ pdf of M_1: the IMF """

    proinitial_mass_max = K1*x**(-alpha_IMF)
    return proinitial_mass_max


def prior_loga(x):
    """ pdf of a: flat in log """
    prob_loga = 1./(semi_major_axis_max-semi_major_axis_min)
    return prob_loga

def prior_q(x):
    """ pdf of q: uniform """
    promass_ratio_max = 1./(mass_ratio_max-mass_ratio_min)
    return promass_ratio_max


#########




def sampling_from_IMF(Nsamples): 
    '''returns  N samples following the IMF using inverse sampling'''  
    samples = ((np.random.uniform(0,1, Nsamples))*(initial_mass_max**(1.-alpha_IMF) - initial_mass_min**(1.-alpha_IMF)) +initial_mass_min**(1-alpha_IMF))**(1./(1.-alpha_IMF))
    return samples


def sampling_from_a(Nsamples):
    '''returns  N samples distributed log uniform using inverse sampling'''
    return 10**np.random.uniform(semi_major_axis_min, semi_major_axis_max, Nsamples)


def sampling_from_q(Nsamples):
    '''returns  N samples distributed uniformly '''
    return np.random.uniform(mass_ratio_min, mass_ratio_max, Nsamples)





def calculateFractionExploration(fexploration, frefinement):
    """ calculates the fraction of samples from the exploration phase
        see also eq. 12 in https://arxiv.org/pdf/1905.00910.pdf """
    

    Nexploration = len(fexploration['systems/weight'])
    Nrefinement  = len(frefinement['systems/weight'])

    fexpl = Nexploration / (Nexploration + Nrefinement) 

    return fexpl


# for ind, outputDir in enumerate([outputDirectoryExpl, outputDirectoryRef]):
fexploration = h5.File(os.path.join(outputDirectoryExpl,'COMPASOutput.h5'), 'r+') # get COMPAS output
frefinement  = h5.File(os.path.join(outputDirectoryRef, 'COMPASOutput.h5'), 'r+') 



def obtaincombineddataSTROOPWAFEL(datafile,  paramHeader, outputDirectoryExpl, outputDirectoryRef):
    '''
    datafile # datafile that contains wanted parameter, e.g. 'systems' or 'doubleCompactObjects'
    paramHeader # header name of the wanted parameter in the'systems' or e.g. 'doubleCompactObjects' file
    outputDirectoryExpl, outputDirectoryRef # paths to exploration & refinement phase data
    '''

    dataexploration = h5.File(os.path.join(outputDirectoryExpl,'COMPASOutput.h5'), 'r+')[datafile + '/' + paramHeader] # get COMPAS output
    datarefinement  = h5.File(os.path.join(outputDirectoryRef, 'COMPASOutput.h5'), 'r+')[datafile + '/' + paramHeader] 

    # combine data and weights from exploration and refinement                     
    combineddata = np.asarray(np.concatenate((dataexploration[...].squeeze(), datarefinement[...].squeeze())))

    param = combineddata
    
    return param




Rcoeff =\
    np.asarray([[1.71535900,    0.62246212,     -0.92557761,    -1.16996966,    -0.30631491],\
    [6.59778800,    -0.42450044,    -12.13339427,   -10.73509484,   -2.51487077],\
    [10.08855000,   -7.11727086,    -31.67119479,   -24.24848322,   -5.33608972],\
    [1.01249500,    0.32699690,     -0.00923418,    -0.03876858,    -0.00412750],\
    [0.07490166,    0.02410413,     0.07233664,     0.03040467,     0.00197741],\
    [0.01077422,    0.00000000,     0.00000000,     0.00000000,     0.00000000],\
    [3.08223400,    0.94472050,     -2.15200882,    -2.49219496,    -0.63848738],\
    [17.84778000,   -7.45345690,    -48.96066856,   -40.05386135,   -9.09331816],\
    [0.00022582,    -0.00186899,    0.00388783,     0.00142402,     -0.00007671]])

    
RsolToAU = 0.00465047 #    # // Solar radius (in AU)

                
def calculateRadiusCoefficient(whichCoefficient, logMetallicityXi):
    """
    /*
     Calculate an alpha-like metallicity dependent constant for the radius
     
     Equation 4 in Tout et al 1996
     
     Parameters
     -----------
     whichCoefficient : int
     Which coefficient to evaluate:
     COEFF_THETA     = 0
     COEFF_IOTA      = 1
     COEFF_KAPPA     = 2
     COEFF_LAMBDA    = 3
     COEFF_MU        = 4
     COEFF_NU        = 5
     COEFF_XI        = 6
     COEFF_OMICRON   = 7
     COEFF_PI        = 8
     
     logMetallicityXi:
     log10(Metallicity / Zsol) where Metallicity = Z (Z = 0.02 = Zsol)
     
     Returns
     ----------
     alpha : float
     Constant alpha

     */
    """
    
    a = Rcoeff[whichCoefficient][0]
    b = Rcoeff[whichCoefficient][1]
    c = Rcoeff[whichCoefficient][2]
    d = Rcoeff[whichCoefficient][3]
    e = Rcoeff[whichCoefficient][4]

  
    return a + b * logMetallicityXi + c * np.power(logMetallicityXi, 2.0) + d * np.power(logMetallicityXi, 3.0) + e * np.power(logMetallicityXi, 4.0)



#// Calculate metallicity dependent radius coefficients
def    calculateAllRadiusCoefficients(logMetallicityXi):

    """
    Calculate metallicity dependent radius coefficients at start up.

    See calculateRadiusCoefficient

    Parameters
    -----------
    logMetallicityXi : double
        log of metallicity

    Returns
    --------
    """
    
    COEFF_THETA     = 0
    COEFF_IOTA      = 1
    COEFF_KAPPA     = 2
    COEFF_LAMBDA    = 3
    COEFF_MU        = 4
    COEFF_NU        = 5
    COEFF_XI        = 6
    COEFF_OMICRON   = 7
    COEFF_PI        = 8    
    
    
    radius_coefficient_theta   = calculateRadiusCoefficient(COEFF_THETA,   logMetallicityXi);
    radius_coefficient_iota    = calculateRadiusCoefficient(COEFF_IOTA,    logMetallicityXi);
    radius_coefficient_kappa   = calculateRadiusCoefficient(COEFF_KAPPA,   logMetallicityXi);
    radius_coefficient_lamda   = calculateRadiusCoefficient(COEFF_LAMBDA,  logMetallicityXi);
    radius_coefficient_mu      = calculateRadiusCoefficient(COEFF_MU,      logMetallicityXi);
    radius_coefficient_nu      = calculateRadiusCoefficient(COEFF_NU,      logMetallicityXi);
    radius_coefficient_xi      = calculateRadiusCoefficient(COEFF_XI,      logMetallicityXi);
    radius_coefficient_omicron = calculateRadiusCoefficient(COEFF_OMICRON, logMetallicityXi);
    radius_coefficient_Pi      = calculateRadiusCoefficient(COEFF_PI,      logMetallicityXi);    
    
    
    

    return radius_coefficient_theta, \
                radius_coefficient_iota, radius_coefficient_kappa,\
                radius_coefficient_lamda, radius_coefficient_mu, \
                radius_coefficient_nu, radius_coefficient_xi, \
                radius_coefficient_omicron, radius_coefficient_Pi
                


   


##########################################


def inverse_sampling_from_power_law(Nprior, power, xmax, xmin):
    # // This function draws samples from a power law distribution p(x) ~ x^(n) between xmin and xmax
    # // Checked against python code for the same function
    
    u = np.random.rand(Nprior)              #// Draw N random number between 0 and 1
    
    if(power == -1.0):
        return np.exp(u*np.log(xmax/xmin))*xmin
    
    else:
        return np.power((u*(np.power(xmax, power+1.0) - np.power(xmin, power+1.0)) + np.power(xmin, power+1.0)), 1.0/(power+1.0))



def sampling_from_e(Nprior, eccentricity_distribution, eMax, eMin):


    if eccentricity_distribution=='ZERO':
        m_Eccentricity =   np.zeros(Nprior) 

    elif(eccentricityDistribution=="FLAT"):
        ePower = 0
        m_Eccentricity = inverse_sampling_from_power_law(Nprior, ePower, eMax, eMin)
    
    elif((eccentricityDistribution=="THERMALISED") or (eccentricityDistribution=="THERMAL")):
        # Thermal eccentricity distribution p(e) = 2e
        ePower = 1
        m_Eccentricity = inverse_sampling_from_power_law(Nprior, ePower, eMax, eMin)
    else:
        print('Error: your chosen eccentricityDistribution, ',eccentricity_distribution, ' ,does not exist, =')

    return m_Eccentricity



def  RadiusZAMS(MZAMS, radius_coefficient_theta, \
                radius_coefficient_iota, radius_coefficient_kappa,\
                radius_coefficient_lamda, radius_coefficient_mu, \
                radius_coefficient_nu, radius_coefficient_xi, \
                radius_coefficient_omicron, radius_coefficient_Pi):
    """Calculate Zero Age Main Sequence (ZAMS) Radius  using Pols et al formulae
        Calculate radius at ZAMS in units of Rsol

         Equation 2 in Tout et al 1996

         Parameters
         -----------
         MZAMS : float
            Zero age main sequence mass in Msol

         Returns
         --------
         RZAMS : float
         Radius in units of Rsol
     
     """



    top     = radius_coefficient_theta * np.power(MZAMS,(2.5)) +\
    radius_coefficient_iota * np.power(MZAMS,(6.5)) + \
    radius_coefficient_kappa * np.power(MZAMS,(11.0)) + \
    radius_coefficient_lamda * np.power(MZAMS,(19.0)) +\
    radius_coefficient_mu * np.power(MZAMS,(19.5))

    bottom  = radius_coefficient_nu + radius_coefficient_xi * np.power(MZAMS,(2.0)) + radius_coefficient_omicron * np.power(MZAMS,(8.5)) + np.power(MZAMS,(18.5)) + radius_coefficient_Pi * np.power(MZAMS,(19.5))
    
    RadiusZAMS = top/bottom 
    
    return RadiusZAMS





def rocheLobeRadius(Mass_i, Mass_j):
    """
     Calculate the roche lobe radius using the chosen approximation
     
     By default uses the fit given in Equation 2 of Eggleton 1983 (Could in principle pass it options and use other fits like Sepinsky+)
     
     Parameters
     -----------
     Mass_i : double
        Primary mass
     Mass_j : double
        Secondary mass
     
     Returns
     --------
     R_L / a : double
     Radius of Roche lobe in units of the semi-major axis a

     """

    q = Mass_i/Mass_j
    top = 0.49;
    bottom = 0.6 + np.power(q, -2.0/3.0) * np.log(1.0 + np.power(q, 1.0/3.0))

    return top/bottom;




def CalculateFractionRejectedPriors(eccentricity_distribution, eMax, eMin, minimum_secondary_mass):
    """Calcultes the fraction of samples that falls outside of the parameter space, 
    because of rejection sampling by either RLOFonZAMS or M2 < minMass in COMPAS
    """

    # monte carlo simulation with resolution 1E8 samples
    sum_rejected, NPrior = 0, int(1E8) 

    samples_M1 = sampling_from_IMF(NPrior)  
    samples_a = sampling_from_a(NPrior)
    samples_q = sampling_from_q(NPrior) 

    
    
    ########## CALCULATE MASK RLOFonZAMS DETAILED:########################
    samples_M2 = samples_q*samples_M1
    m_R1ZAMS             = RadiusZAMS(samples_M1, radius_coefficient_theta, radius_coefficient_iota, \
                                     radius_coefficient_kappa, radius_coefficient_lamda, \
                                     radius_coefficient_mu, radius_coefficient_nu, \
                                     radius_coefficient_xi, radius_coefficient_omicron,\
                                     radius_coefficient_Pi)
    m_R2ZAMS             = RadiusZAMS(samples_M2, radius_coefficient_theta, radius_coefficient_iota, \
                                     radius_coefficient_kappa, radius_coefficient_lamda, \
                                     radius_coefficient_mu, radius_coefficient_nu, \
                                     radius_coefficient_xi, radius_coefficient_omicron,\
                                     radius_coefficient_Pi)


    m_Eccentricity = sampling_from_e(NPrior, eccentricity_distribution, eMax, eMin)
    m_SemiMajorAxis=   samples_a # sampled MC 

    m_RadiusRL1             = rocheLobeRadius(samples_M1, samples_M2);
    m_RadiusRL2             = rocheLobeRadius(samples_M2, samples_M1);


    rocheLobeTracker1 = (m_R1ZAMS*RsolToAU)/(m_SemiMajorAxis*(1.0 - m_Eccentricity)*m_RadiusRL1)
    rocheLobeTracker2 = (m_R2ZAMS*RsolToAU)/(m_SemiMajorAxis*(1.0 - m_Eccentricity)*m_RadiusRL2)


    
    maskRLOFonZAMS1 = (rocheLobeTracker1 > 1.0) 
    maskRLOFonZAMS2 =   (rocheLobeTracker2 > 1.0)
    maskRLOFonZAMS = (rocheLobeTracker1 > 1.0) |  (rocheLobeTracker2 > 1.0)


    
    # mask samples that are inside parameter space     
    # m2 with too low masses are rejected within COMPAS
    # approximate property to not have RLOFonZAMS
    # mask samples that we keep
    mask = (((samples_q*samples_M1)>= minimum_secondary_mass) \
    & (maskRLOFonZAMS==0) )

    sum_rejected = len(mask) - np.sum(mask)   # total rejected samples of all gaussians


    # calculate the fraction of samples that is rejected
    Prej = sum_rejected / (NPrior)


    return Prej




def CalculateFractionRejectedGaussians(masterDirectory, eccentricity_distribution, eMax, eMin, minimum_secondary_mass): #, minSeparationRLOFonZAMS):
    """Calcultes the fraction of samples that falls outside of the parameter space, F_rej
    when sampling from the Gaussians q(x) (see also Eq. 8 and 9 in https://arxiv.org/pdf/1905.00910.pdf)

    """


    d_gaussians = np.genfromtxt(masterDirectory + '/STROOPWAFELgaussians.txt', skip_header=2)
    targetmeans = d_gaussians[:,0:3]
    targetstd   = d_gaussians[:,3: ]

    sum_rejected, NperGauss = 0, 10000

    std_m1, std_a, std_q =  np.transpose(targetstd) # obtain std (\sigma) from AISgaussians.txt
    mu_m1, mu_a, mu_q = np.transpose(targetmeans)

    # for all Gaussians count how many samples fall outside of parameter space
    for i in range(len(targetmeans)):
        # sample NperGauss samples from i-th Gaussian for M_1, a and q. To the power 2 to obtain sigma^2 instead of std (sigma)
        samples_M1 = (multivariate_normal.rvs(mean = mu_m1[i], cov = std_m1[i]**2, size = NperGauss))  
        samples_a = 10**(multivariate_normal.rvs(mean = mu_a[i], cov = std_a[i]**2, size = NperGauss)) 
        samples_q = multivariate_normal.rvs(mean = mu_q[i], cov = std_q[i]**2, size = NperGauss)  

        # mask samples that are inside parameter space  
       # m2 with too low masses are rejected within COMPAS
        # approximate property to not have RLOFonZAMS
        

        ########## CALCULATE MASK RLOFonZAMS:########################
        samples_M2 = samples_q*samples_M1
        m_R1ZAMS             = RadiusZAMS(samples_M1, radius_coefficient_theta, radius_coefficient_iota, \
                                         radius_coefficient_kappa, radius_coefficient_lamda, \
                                         radius_coefficient_mu, radius_coefficient_nu, \
                                         radius_coefficient_xi, radius_coefficient_omicron,\
                                         radius_coefficient_Pi)
        m_R2ZAMS             = RadiusZAMS(samples_M2, radius_coefficient_theta, radius_coefficient_iota, \
                                         radius_coefficient_kappa, radius_coefficient_lamda, \
                                         radius_coefficient_mu, radius_coefficient_nu, \
                                         radius_coefficient_xi, radius_coefficient_omicron,\
                                         radius_coefficient_Pi)


        m_Eccentricity =   sampling_from_e(NperGauss, eccentricity_distribution, eMax, eMin)
        m_SemiMajorAxis=   samples_a # sampled MC 

        m_RadiusRL1             = rocheLobeRadius(samples_M1, samples_M2);
        m_RadiusRL2             = rocheLobeRadius(samples_M2, samples_M1);


        rocheLobeTracker1 = (m_R1ZAMS*RsolToAU)/(m_SemiMajorAxis*(1.0 - m_Eccentricity)*m_RadiusRL1)
        rocheLobeTracker2 = (m_R2ZAMS*RsolToAU)/(m_SemiMajorAxis*(1.0 - m_Eccentricity)*m_RadiusRL2)

        maskRLOFonZAMS = (rocheLobeTracker1 > 1.0) |  (rocheLobeTracker2 > 1.0)
        
       
        mask = ((samples_M1 >= initial_mass_min) & (samples_M1 <= initial_mass_max) & \
         (samples_a >= (10.**semi_major_axis_min)) &  (samples_a <= (10.**semi_major_axis_max)) \
                    &  (samples_q >= mass_ratio_min) & (samples_q <= mass_ratio_max) \
                    & ((samples_q*samples_M1)>= minimum_secondary_mass) \
                    & (maskRLOFonZAMS == 0 ) )
                
                
        Nrejected = len(mask) - np.sum(mask) 
        sum_rejected += Nrejected  # total rejected samples of all gaussians

    # calculate the nr of samples total drawn from Gaussians
    NsamplesFromGaussians = NperGauss * len(targetmeans) 
    # calculate the fraction of samples that is rejected
    fraction_rejected = sum_rejected / (NsamplesFromGaussians)

    return fraction_rejected



def calculatessmallq(array, masterDirectory):  #new version
    '''function that calculates the pdf of the instrumental distribution
    input is an array of samples, array of AISmeans and AIScov, 
    it returns the probability for each of the samples for the mixture pdf
    this is Equation 8 (q(x)) of https://arxiv.org/pdf/1905.00910.pdf
    '''
    
    
    d_gaussians = np.genfromtxt(masterDirectory + '/STROOPWAFELgaussians.txt', skip_header=2)
    targetmeans, targetstd = d_gaussians[:,0:3], d_gaussians[:,3: ]
    Ngaussians = len(targetmeans)

    # make a matrix which we will fill up with contributions to the PDF from each Gaussian
    xPDF = np.zeros((Ngaussians, len(array)))
    
    
    # calculate for each gaussian the probability of the array \
    # then sum all the probabilities together and normalize
    for i in range(Ngaussians):
        COV_M1, COV_a, COV_q =  targetstd[i]**2 # this is \sigma^2 in definition \Sigma, we need to take **2 since the Gaussians feed to COMPAS in c++ are np.sqrt(factor)*cov (so cov instead of cov^2)
        COV = [[COV_M1, 0, 0], [0, COV_a, 0], [0, 0, COV_q]]
        # print len(array), len(targetmeans[i]), len(COV)
        # print 'array', array
        # print 'mean', targetmeans[i]
        # print 'cov' , COV
        xPDF[i, :] = (multivariate_normal.pdf(list(array), targetmeans[i], COV))
    
    qPDF = np.sum(xPDF, axis = 0)  * (float(Ngaussians))**-1  # Eq. 8 of https://arxiv.org/pdf/1905.00910.pdf
    

    return qPDF




def calculatesPi(array):  
    ''' returns the probability for each of the samples for the priors
    see Eq. 1 of https://arxiv.org/pdf/1905.00910.pdf 
    '''
        
    # pdf contribution from prior sampling
    priorM1 =  prior_M1(np.transpose(array)[0])
    priora =  prior_loga(np.transpose(array)[1])
    priorq =  prior_q(np.transpose(array)[2])
    
    piPDF= priorM1 * priora * priorq 
    
    
    return piPDF


def calculateMixtureWeights(pi_x, q_x,  PIrej, Qrej, fexpl):
    """  """
    #normalization factors for pi(x) and q(x) due to rejection sampling COMPAS
    PInorm =  1./(1.-PIrej) 
    Qnorm =  1./(1.-Qrej) 
    
    numerator =  PInorm * pi_x
    denomenator = (fexpl * PInorm * pi_x)   +  ( (1.-fexpl) * Qnorm * q_x) 
    
    mixtureweights = numerator / denomenator
    
    return mixtureweights


# calculate fraction of samples in exploration phase 
fexpl = calculateFractionExploration(fexploration=fexploration, frefinement=frefinement) 

# obtain combined data of prior parameters:
m1zamsDCO  = obtaincombineddataSTROOPWAFEL(datafile='doubleCompactObjects', paramHeader='M1ZAMS', outputDirectoryExpl=outputDirectoryExpl, outputDirectoryRef=outputDirectoryRef)
m1zamsSystems  = obtaincombineddataSTROOPWAFEL(datafile='systems', paramHeader='mass1', outputDirectoryExpl=outputDirectoryExpl, outputDirectoryRef=outputDirectoryRef)
qDCO  = obtaincombineddataSTROOPWAFEL(datafile='doubleCompactObjects', paramHeader='M2ZAMS', outputDirectoryExpl=outputDirectoryExpl, outputDirectoryRef=outputDirectoryRef)/m1zamsDCO
qSystems  = obtaincombineddataSTROOPWAFEL(datafile='systems', paramHeader='mass2', outputDirectoryExpl=outputDirectoryExpl, outputDirectoryRef=outputDirectoryRef)/m1zamsSystems
logaDCO   = np.log10(obtaincombineddataSTROOPWAFEL(datafile='doubleCompactObjects', paramHeader='separationInitial', outputDirectoryExpl=outputDirectoryExpl, outputDirectoryRef=outputDirectoryRef))
logaSystems   = np.log10(obtaincombineddataSTROOPWAFEL(datafile='systems', paramHeader='separation', outputDirectoryExpl=outputDirectoryExpl, outputDirectoryRef=outputDirectoryRef))



samplesDCO = np.transpose([m1zamsDCO, logaDCO, qDCO])
samplesSystems = np.transpose([m1zamsSystems, logaSystems, qSystems])





##############################################
# below functions are needed for rejection sampling normalization RLOFonZAMS
# all functions come from COMPAS .cpp files, and are in COMPAS used to determine whether a binary has RLOFonZAMS



logMetallicityXi = np.log10(programOptions.metallicity) 
radius_coefficient_theta, \
    radius_coefficient_iota, radius_coefficient_kappa, \
    radius_coefficient_lamda, radius_coefficient_mu, \
    radius_coefficient_nu, radius_coefficient_xi, \
    radius_coefficient_omicron, radius_coefficient_Pi  \
    = calculateAllRadiusCoefficients(logMetallicityXi)



# Monte Carlo estimate the fraction of samples that COMPAS rejects from the priors    
Prej = CalculateFractionRejectedPriors(eccentricity_distribution, eMax, eMin, minimum_secondary_mass) 

# Monte Carlo estimate the fraction of samples that COMPAS rejects from the mixture of Gaussians 
Qrej  = CalculateFractionRejectedGaussians(masterDirectory, eccentricity_distribution, eMax, eMin, minimum_secondary_mass)


piDCO = calculatesPi(array=samplesDCO) # unnormalized
qDCO  = calculatessmallq(array=samplesDCO, masterDirectory=masterDirectory) # unnormalized
weightsDCO = calculateMixtureWeights(pi_x=piDCO, q_x=qDCO, PIrej=Prej, Qrej=Qrej, fexpl=fexpl) # normalized weights (including rejection sampling)
print('DCO weights are calculated')


piSystems = calculatesPi(array=samplesSystems) # unnormalized
qSystems  = calculatessmallq(array=samplesSystems, masterDirectory=masterDirectory) # unnormalized
weightsSystems = calculateMixtureWeights(pi_x=piSystems, q_x=qSystems,  PIrej=Prej, Qrej=Qrej, fexpl=fexpl) # normalized
print('Systems weights are calculated')


# normalization factor, so that sum(weightsSystems) = 1
normFactor = len(weightsSystems) / np.sum(weightsSystems)
# normalize weights
weightsDCO=weightsDCO*normFactor
weightsSystems=weightsSystems*normFactor

# obtain Number of systems  and DCOs in exploration
NexplorationSystems = len(fexploration['systems/weight'])
NexplorationDCO  = len(fexploration['doubleCompactObjects/weight'])



weightsDCOStroopwafelList = [weightsDCO[0:NexplorationDCO], weightsDCO[NexplorationDCO:]]
weightsSystemsStroopwafelList = [weightsSystems[0:NexplorationSystems], weightsSystems[NexplorationSystems:]]



# for both directories, change default weights to weights above
for ind, outputDir in enumerate([outputDirectoryExpl, outputDirectoryRef]):
    f = h5.File(os.path.join(outputDir,'COMPASOutput.h5'), 'r+') # get COMPAS output

    # get DCO weights from Stroopwafel (created above)
    weightsDCOStroopwafel = weightsDCOStroopwafelList[ind] 
    weightsSystemsStroopwafel = weightsSystemsStroopwafelList[ind]

    #get weights in DCO and systems file (are default 1)
    weightsDCOs = f['doubleCompactObjects/weight']
    weightsSystems = f['systems/weight']

    # overwrite weights DCO and Systems in hdf5 files with weights from Stroopwafel sampling
    weightsDCOs[...] = np.asarray([weightsDCOStroopwafel]).T
    weightsSystems[...] = np.asarray([weightsSystemsStroopwafel]).T



    ## DO THE SAME FOR SAMPLING PHASE PARAMETER ## 
    #exploratory phase
    if ind==0:
        samplingPhaseDCOStroopwafel = np.ones_like(weightsDCOStroopwafelList[ind]) 
        samplingPhaseSystemsStroopwafel = np.ones_like(weightsSystemsStroopwafelList[ind])
    else:
        samplingPhaseDCOStroopwafel =  np.full_like(weightsDCOStroopwafel[ind],int(2)) 
        samplingPhaseSystemsStroopwafel = np.full_like(weightsSystemsStroopwafel[ind], int(2))

    #get weights in DCO and systems file (are default -1)
    samplingPhaseDCOs = f['doubleCompactObjects/samplingPhase']
    samplingPhaseSystems = f['systems/samplingPhase']

    # overwrite samplingphasrParameter DCO and Systems in hdf5 files with weights from Stroopwafel sampling
    samplingPhaseDCOs[...] = np.asarray([samplingPhaseDCOStroopwafel]).T
    samplingPhaseSystems[...] = np.asarray([samplingPhaseSystemsStroopwafel]).T



print('finished postProcessing STROOPWAFEL weights')