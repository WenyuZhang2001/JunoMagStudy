
import pandas as pd

import CoordinateTransform
import Juno_Mag_MakeData_Function
import os
import numpy as np
from scipy.linalg import lstsq
from scipy.special import lpmn,factorial
from sklearn.linear_model import Ridge
import joblib



def B_In_obs_Calculate(data,B_Ex):
    B_In_obs = pd.DataFrame(columns=['Br','Btheta','Bphi','Btotal'])
    B_In_obs['Br'] = data['Br'] - B_Ex['Br']
    B_In_obs['Btheta'] = data['Btheta'] - B_Ex['Btheta']
    B_In_obs['Bphi'] = data['Bphi'] - B_Ex['Bphi']
    B_In_obs['Btotal'] = np.sqrt(B_In_obs['Br']**2 + B_In_obs['Btheta']**2 + B_In_obs['Bphi']**2)
    return B_In_obs

def Model_Simulation(data, B_In_obs, NMAX=10, NMIN=1, SVD_On=True,LSTSQ_On = True, Regularization_On = True,
                     SVD_rcond= 1e-15,Regularize_lambda=1,Regularize_rc=1,path='Spherical_Harmonic_Model'):

    os.makedirs(path, exist_ok=True)

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
        A = Schmidt_Matrix(data, Nmax)


        if LSTSQ_On:
            # Solve the least squares problem
            gnm_hnm, residuals, rank, s = lstsq(A, B)

            np.save(f'{path}/Inversion_LSTSQ_coefficients_gnm_hnm_Nmax{Nmax}.npy',gnm_hnm)
            np.save(f'{path}/Inversion_LSTSQ_coefficients_rank_Nmax{Nmax}.npy',rank)
            print(f'The LSTSQ number of the gnm & hnm is {num_coeffs}')

        if SVD_On:

            U, sigma, VT = np.linalg.svd(A, full_matrices=False)
            A_inv = np.linalg.pinv(A, rcond=SVD_rcond)
            gnm_hnm_SVD = np.dot(A_inv, B)

            np.save(f'{path}/Inversion_SVD_coefficients_gnm_hnm_Nmax{Nmax}.npy', gnm_hnm_SVD)
            np.save(f'{path}/Inversion_SVD_coefficients_U_Nmax{Nmax}.npy', U)
            np.save(f'{path}/Inversion_SVD_coefficients_S_Nmax{Nmax}.npy', sigma)
            np.save(f'{path}/Inversion_SVD_coefficients_V_Nmax{Nmax}.npy', VT)

            print(f'The SVD Shape of the gnm_hnm is {gnm_hnm_SVD.shape}')
            print(f'The SVD Spape of U is {U.shape}, S is {sigma.shape}, V is {VT.shape}')
            print(f'SVD Nmax={Nmax} finished')
            print('-'*50)
        if Regularization_On:

            Regularized_A = Schmidt_Matrix_Regularization(A,Nmax=Nmax,rc=Regularize_rc,lambda_reg=Regularize_lambda)
            U, sigma, VT = np.linalg.svd(Regularized_A, full_matrices=False)
            A_inv = np.linalg.pinv(Regularized_A, rcond=SVD_rcond)
            New_B = np.dot(A.T, B)
            gnm_hnm_Regularized = np.dot(A_inv, New_B)


            np.save(f'{path}/Inversion_Regularized_SVD_coefficients_gnm_hnm_Nmax{Nmax}.npy', gnm_hnm_Regularized)
            np.save(f'{path}/Inversion_Regularized_SVD_coefficients_U_Nmax{Nmax}.npy', U)
            np.save(f'{path}/Inversion_Regularized_SVD_coefficients_S_Nmax{Nmax}.npy', sigma)
            np.save(f'{path}/Inversion_Regularized_SVD_coefficients_V_Nmax{Nmax}.npy', VT)
            print(f'The Regularized SVD Shape of the gnm_hnm is {gnm_hnm_Regularized.shape}')
            print(f'The Regularized SVD Spape of U is {U.shape}, S is {sigma.shape}, V is {VT.shape}')

            print(f'Regularized SVD Nmax={Nmax} finished')
            print('-' * 50)
            # ridge_model = Ridge(alpha=Ridge_alpha)
            # ridge_model.fit(A, B)
            # joblib.dump(ridge_model, f'{path}/ridge_model_Nmax{Nmax}_Alpha{Ridge_alpha}.joblib')
            # print("Ridge Model Coefficients:", ridge_model.coef_.shape)
            # print(f'Ridged Alpha={Ridge_alpha}')


        print(f'Loop Nmax={Nmax} End')

        if (SVD_On or Regularization_On or LSTSQ_On) == False:
            print('No Model Trained!')
        print('=' * 50)

    data['theta'] = data['theta'] * 360 / (2*np.pi)
    data['phi'] = data['phi']  * 360 / (2*np.pi)


def read_gnm_hnm_data(method='SVD', Nmax=13, path='Spherical_Harmonic_Model'):

    gnm_hnm_coeffi = np.load(f'{path}/Inversion_{method}_coefficients_gnm_hnm_Nmax{Nmax}.npy')

    return gnm_hnm_coeffi

def Schmidt_Matrix(data,Nmax):
    # Initialize the design matrix A
    num_coeffs = int((Nmax + 2) * Nmax)
    print(f'Schmidt Coefficient total numbers = {num_coeffs}\ngnm_num={(Nmax+3)*Nmax/2} hnm_num={(Nmax+3)*Nmax/2-Nmax}')
    A = np.zeros((len(data)*3, num_coeffs))

    # Function to calculate the Schmidt semi-normalization factor
    def schmidt_semi_normalization(n, m):
        return ((-1)**m)*np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))

    # Populate the design matrix A
    for i, (r_val, theta_val, phi_val) in enumerate(zip(data['r'], data['theta'], data['phi'])):
        for n in range(1,Nmax + 1):
            for m in range(n + 1):
                P, dP = lpmn(m, n, np.cos(theta_val))
                N_lm = schmidt_semi_normalization(n, m)

                # gnm index
                # (n-1+3)*(n-1)/2 + m
                gnm_index = int((n+2)*(n-1)/2 + m)
                # hnm index
                # (n-1+3)*(n-1)/2 - (n-1) + m-1 + gnm_num (= (n+3)*n/2
                hnm_index = int((n+2)*(n-1)/2-(n-1)+m-1 + (Nmax+3)*Nmax/2)
                # Contribution to Br from gnm
                A[3*i, gnm_index] = (n + 1) * (r_val**(-n - 2)) * np.cos(m * phi_val) * P[m, n] * N_lm
                # Contribution to Btheta from gnm
                A[3*i + 1, gnm_index] = -(r_val**(-n - 2)) * np.cos(m * phi_val) * (-np.sin(theta_val)) * dP[m, n] * N_lm
                # Contribution to Bphi from gnm
                A[3*i + 2, gnm_index] = m * (r_val**(-n - 2)) * np.sin(m * phi_val) * P[m, n] * N_lm / np.sin(theta_val)

                if m==0:
                    continue
                # Contribution to Br from hnm
                A[3 * i, hnm_index] = (n + 1) * (r_val ** (-n - 2)) * np.sin(m * phi_val) * P[m, n] * N_lm
                # Contribution to Btheta from hnm
                A[3 * i + 1, hnm_index] = -(r_val ** (-n - 2)) * np.sin(m * phi_val) * (-np.sin(theta_val)) * \
                                                   dP[m, n] * N_lm
                # Contribution to Bphi from hnm
                A[3*i + 2, hnm_index] = m * (r_val**(-n - 2)) * (-np.cos(m * phi_val)) * P[m, n] * N_lm / np.sin(theta_val)
    print(f'SchmidtMatrix calculate success. Shape = {A.shape}\n'+'+'*50)

    return A

def calculate_Bfield(data,path='Spherical_Harmonic_Model',Nmax=10,method='SVD',Rc=1.0):
    '''

    :param data:  data [theta] and [phi] is in degree, this function will auto trans it to rad and trans back at the end
    :param path:
    :param Nmax:
    :param method: SVD,Regularized_SVD
    :param Ridge_alpha:
    :return:
    '''
    data['theta'] = data['theta'] / 360 * 2 * np.pi
    data['phi'] = data['phi'] / 360 * 2 * np.pi

    SchmidtMatrix = Schmidt_Matrix(data,Nmax)
    # ridge_model = joblib.load(f'{path}/ridge_model_Nmax{Nmax}_Alpha{Ridge_alpha}.joblib')
    # B_Model = ridge_model.predict(SchmidtMatrix)
    gnm_hnm_coeffi = read_gnm_hnm_data(path=path,Nmax=Nmax,method=method)
    ParameterScale(gnm_hnm_coeffi,Nmax=Nmax,Rc=Rc)
    B_Model = np.dot(SchmidtMatrix,gnm_hnm_coeffi)

    B_Model = B_Model.reshape((int(len(B_Model)/3),3))
    B_Model_df = pd.DataFrame(B_Model,columns=['Br','Btheta','Bphi'],index=data['X'].index)


    B_Model_df['Btotal'] = np.sqrt(B_Model_df['Br']**2 + B_Model_df['Btheta']**2 + B_Model_df['Bphi']**2)
    # if method == 'Ridge':
    #     B_Model_df['alpha'] = ridge_model.alpha * np.ones(len(B_Model_df))

    print(f'B Field of Model Calculated \n Nmax={Nmax}')

    data['theta'] = data['theta'] * 360 / (2 * np.pi)
    data['phi'] = data['phi'] * 360 / (2 * np.pi)

    Bx, By, Bz = CoordinateTransform.SphericaltoCartesian_Bfield(data['r'].to_numpy(),
                                                                 data['theta'].to_numpy(),
                                                                 data['phi'].to_numpy(),
                                                                 B_Model_df['Br'].to_numpy(),
                                                                 B_Model_df['Btheta'].to_numpy(),
                                                                 B_Model_df['Bphi'].to_numpy())

    B_Model_df['Bx'] = Bx
    B_Model_df['By'] = By
    B_Model_df['Bz'] = Bz


    return B_Model_df

def ParameterScale(gnm_hnm_coeffi,Nmax,Rc = 1.0):
    for n in range(1, Nmax + 1):
        for m in range(n + 1):
            # gnm index
            # (n-1+3)*(n-1)/2 + m
            gnm_index = int((n + 2) * (n - 1) / 2 + m)
            # hnm index
            # (n-1+3)*(n-1)/2 - (n-1) + m-1 + gnm_num (= (n+3)*n/2
            hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)

            Scale = Rc**(n-1)

            gnm_hnm_coeffi[gnm_index] = gnm_hnm_coeffi[gnm_index] * Scale
            gnm_hnm_coeffi[hnm_index] = gnm_hnm_coeffi[hnm_index] * Scale

    return

def Schmidt_Matrix_Regularization(A,Nmax,rc,lambda_reg=1):

    num_coeffs = int((Nmax + 2) * Nmax)
    diagonal_elements = np.zeros(num_coeffs)

    # Calculate the diagonal elements of the regularization matrix Gamma
    for n in range(1,Nmax+1):
        for m in range(n+1):
            gnm_index = int((n + 2) * (n - 1) / 2 + m)
            hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)

            diagonal_elements[gnm_index] = (n + 1) * (1 / rc) ** (2 * n + 4)
            diagonal_elements[hnm_index] = (n + 1) * (1 / rc) ** (2 * n + 4)

    # Construct the regularization matrix Gamma
    Gamma = np.diag(diagonal_elements)

    ATA = np.dot(A.T,A)
    lambdaGammaTGamma = lambda_reg * np.dot(Gamma.T, Gamma)

    # Compute the new matrix to decompose
    modified_matrix = ATA + lambdaGammaTGamma

    return modified_matrix

