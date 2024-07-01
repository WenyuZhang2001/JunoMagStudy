import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from scipy.special import lpmn,factorial
from sklearn.linear_model import Ridge
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions

# path = 'Spherical_Harmonic_Model/First20_Orbit_Model'
# path = 'Spherical_Harmonic_Model/Ridged_Model'
# path = 'Spherical_Harmonic_Model/First50_Orbit_Model_RegularizationTest'
# # Make Dir
# os.makedirs(path,exist_ok=True)


# Model = 'jrm33'
# Orbits and Doy time
data = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_Data_1s_2h.csv')
B_Ex = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_Ex_1s_2h.csv')
# B_In = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_In_1s_2h.csv')
# data = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_Data_1s_2h.csv')
# B_Ex = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_Ex_1s_2h.csv')
# B_In = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_In_1s_2h.csv')

# sample it 60s
data = data.iloc[::60]
# B_In = B_In.iloc[::60]
B_Ex = B_Ex.iloc[::60]

# data['r'] = data['r']/0.85

B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data,B_Ex)





# Maximum degree of internal field
# Nmax = 10

# Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In_obs,path=path,NMIN=1,NMAX=21)

# alpha_list = [0.8,1]
# for alpha in alpha_list:
#     Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In_obs,path=path,NMIN=10,NMAX=10,Ridge_alpha=alpha,
#                                                                  LSTSQ_On=False,SVD_On=False)

# Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In_obs,path=path,NMAX=40,NMIN=40,SVD_On=True,LSTSQ_On=False,Ridge_On=False)

Rc_List = [1.0,0.9,0.8,0.7,0.88,0.85,0.92]
# Lambda_List = [1.0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7]
# Lambda_List = [1e-8,1e-9,1e-10,1e-11,1e-12,1e-13]
Lambda_List = [1e-14,1e-15,1e-16,1e-17,1e-18,1e-19,1e-20]
for Rc in Rc_List:
    for Lambda in Lambda_List:

        path = f'Spherical_Harmonic_Model/First50_Orbit_Model_Regularization_{Rc}_{Lambda:.2e}'
        # Make Dir
        os.makedirs(path,exist_ok=True)

        Nmax_list = [1,10,20,30,40]
        for Nmax in Nmax_list:
            Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data, B_In_obs, path=path, NMAX=Nmax, NMIN=Nmax,
                                                                         SVD_On=False, LSTSQ_On=False, Regularization_On=True,Regularize_rc=Rc,Regularize_lambda=Lambda)

