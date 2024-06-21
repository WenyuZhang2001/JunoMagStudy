import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from scipy.special import lpmn,factorial
from sklearn.linear_model import Ridge
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions

path = 'Spherical_Harmonic_Model/First50_Orbit_Model_3GH'
# Make Dir
os.makedirs(path,exist_ok=True)

# Orbits and Doy time
data = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_Data_1s_2h.csv')
B_Ex = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_Ex_1s_2h.csv')
# B_In = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_In_1s_2h.csv')

# sample it 60s
data = data.iloc[::60]
# B_In = B_In.iloc[::60]
B_Ex = B_Ex.iloc[::60]

B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data,B_Ex)

def Model_Simulation(data, B_In_obs, NMAX=10, NMIN=1,SVD_rcond= 1e-15,path='Spherical_Harmonic_Model'):

    data['theta'] = data['theta'] / 360 * 2 * np.pi
    data['phi'] = data['phi'] / 360 * 2 * np.pi

    for Nmax in range(NMIN, NMAX+1):
        # Total number of gnm and hnm coefficients
        num_coeffs = (Nmax + 2) * Nmax

        # Initialize your observations vector B
        B = np.vstack((B_In_obs['Br'], B_In_obs['Btheta'], B_In_obs['Bphi'])).T.reshape(-1)

        # Function to calculate the Schmidt semi-normalization factor

        # Populate the design matrix A

        print(f'The Shape of B Field is {B.shape}')

        # Calculate the Schmidt Matrix
        A = Spherical_Harmonic_InversionModel_Functions.Schmidt_Matrix(data, Nmax)
        A_Br = A[0::3]
        A_Btheta = A[1::3]
        A_Bphi = A[2::3]

        def SVD_Calculate(A,SVD_rcond,path):
            U, sigma, VT = np.linalg.svd(A, full_matrices=False)
            A_inv = np.linalg.pinv(A, rcond=SVD_rcond)
            gnm_hnm_SVD = np.dot(A_inv, B)

            np.save(f'{path}/Inversion_SVD_coefficients_gnm_hnm_Nmax{Nmax}.npy', gnm_hnm_SVD)
            np.save(f'{path}/Inversion_SVD_coefficients_U_Nmax{Nmax}.npy', U)
            np.save(f'{path}/Inversion_SVD_coefficients_S_Nmax{Nmax}.npy', sigma)
            np.save(f'{path}/Inversion_SVD_coefficients_V_Nmax{Nmax}.npy', VT)
            print(f'The SVD Shape of the gnm_hnm is {gnm_hnm_SVD.shape}')
            print(f'The SVD Spahe of U is {U.shape}, S is {sigma.shape}, V is {VT.shape}')

        SVD_Calculate(A_Br,SVD_rcond=SVD_rcond,path=path+'/SVD_Br')
        SVD_Calculate(A_Btheta,SVD_rcond=SVD_rcond,path=path+'/SVD_Btheta')
        SVD_Calculate(A_Bphi,SVD_rcond=SVD_rcond,path=path+'/SVD_Bphi')

        print(f'Loop Nmax={Nmax} End')
        print('=' * 50)

    data['theta'] = data['theta'] * 360 / (2*np.pi)
    data['phi'] = data['phi']  * 360 / (2*np.pi)

