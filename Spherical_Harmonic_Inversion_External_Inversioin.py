import pandas as pd
import os
import numpy as np
from scipy.special import lpmn,factorial

import CoordinateTransform
import Juno_Mag_MakeData_Function
import seaborn as sns
import matplotlib.pyplot as plt

def Model_Simulation(data, B_In_obs, NMAX=10, NMIN=1, SVD_On=True, SVD_rcond= 1e-15,path='Spherical_Harmonic_Model'):

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

        if SVD_On:

            U, sigma, VT = np.linalg.svd(A, full_matrices=False)
            A_inv = np.linalg.pinv(A, rcond=SVD_rcond)
            gnm_hnm_SVD = np.dot(A_inv, B)

            np.save(f'{path}/External_Inversion_SVD_coefficients_gnm_hnm_Nmax{Nmax}.npy', gnm_hnm_SVD)
            np.save(f'{path}/External_Inversion_SVD_coefficients_U_Nmax{Nmax}.npy', U)
            np.save(f'{path}/External_Inversion_SVD_coefficients_S_Nmax{Nmax}.npy', sigma)
            np.save(f'{path}/External_Inversion_SVD_coefficients_V_Nmax{Nmax}.npy', VT)

            print(f'The SVD Shape of the gnm_hnm is {gnm_hnm_SVD.shape}')
            print(f'The SVD Spape of U is {U.shape}, S is {sigma.shape}, V is {VT.shape}')
            print(f'SVD Nmax={Nmax} finished')
            print('-'*50)


        print(f'Loop Nmax={Nmax} End')

        if (SVD_On) == False:
            print('No Model Trained!')
        print('=' * 50)

    data['theta'] = data['theta'] * 360 / (2*np.pi)
    data['phi'] = data['phi']  * 360 / (2*np.pi)

# Function to calculate the Schmidt semi-normalization factor
def schmidt_semi_normalization(n, m):
    return ((-1)**m)*np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))

def Schmidt_Matrix(data,Nmax):
    # Initialize the design matrix A
    num_coeffs = int((Nmax + 2) * Nmax)
    print(f'Schmidt Coefficient total numbers = {num_coeffs}\ngnm_num={(Nmax+3)*Nmax/2} hnm_num={(Nmax+3)*Nmax/2-Nmax}')
    A = np.zeros((len(data)*3, num_coeffs))

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
                A[3*i, gnm_index] = (-n) * (r_val**(n - 1)) * np.cos(m * phi_val) * P[m, n] * N_lm
                # Contribution to Btheta from gnm
                A[3*i + 1, gnm_index] = -(r_val**(n - 1)) * np.cos(m * phi_val) * (-np.sin(theta_val)) * dP[m, n] * N_lm
                # Contribution to Bphi from gnm
                A[3*i + 2, gnm_index] = m * (r_val**(n - 1)) * np.sin(m * phi_val) * P[m, n] * N_lm / np.sin(theta_val)

                if m==0:
                    continue
                # Contribution to Br from hnm
                A[3 * i, hnm_index] = (-n) * (r_val ** (n - 1)) * np.sin(m * phi_val) * P[m, n] * N_lm
                # Contribution to Btheta from hnm
                A[3 * i + 1, hnm_index] = -(r_val ** (n - 1)) * np.sin(m * phi_val) * (-np.sin(theta_val)) * \
                                                   dP[m, n] * N_lm
                # Contribution to Bphi from hnm
                A[3*i + 2, hnm_index] = m * (r_val**(n - 1)) * (-np.cos(m * phi_val)) * P[m, n] * N_lm / np.sin(theta_val)
    print(f'SchmidtMatrix calculate success. Shape = {A.shape}\n'+'+'*50)

    return A

def read_data(year_doy_pj):
    data = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year: [doy[0]]}
            date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

            # read data
            Data = Juno_Mag_MakeData_Function.Read_Data(year_doy, freq=1)
            Data = Data.iloc[::60]

            # 24 hours data
            Time_start = date_list['Time'].iloc[0]
            Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 24

            data_day = Data.loc[Time_start:Time_end]

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    # data.index = data['Time']
    return data
def read_gnm_hnm_data(method='SVD', Nmax=13, path='Spherical_Harmonic_Model'):

    gnm_hnm_coeffi = np.load(f'{path}/External_Inversion_{method}_coefficients_gnm_hnm_Nmax{Nmax}.npy')

    return gnm_hnm_coeffi

def calculate_Bfield(data,path='Spherical_Harmonic_Model',Nmax=10,method='SVD'):

    data['theta'] = data['theta'] / 360 * 2 * np.pi
    data['phi'] = data['phi'] / 360 * 2 * np.pi

    SchmidtMatrix = Schmidt_Matrix(data,Nmax)
    # ridge_model = joblib.load(f'{path}/ridge_model_Nmax{Nmax}_Alpha{Ridge_alpha}.joblib')
    # B_Model = ridge_model.predict(SchmidtMatrix)
    gnm_hnm_coeffi = read_gnm_hnm_data(path=path,Nmax=Nmax,method=method)
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

def calculate_rms_error(B_pred, B_obs):
    return np.sqrt(np.mean((B_pred - B_obs)**2))

def Plot_RMS_Nmax(data,B_Residual,Nmax_List=[1,2,3],path = 'Spherical_Harmonic_Model/First50_Orbit_Model_External',Method='SVD'):

    RMS_df = pd.DataFrame(columns=['Nmax','Br','Btheta','Bphi','Btotal'])

    for Nmax in Nmax_List:
        B_Model_SVD = calculate_Bfield(data, Nmax=Nmax, path=path,method=Method)

        new_row = {'Nmax':Nmax,
                   'Br':calculate_rms_error(B_Model_SVD['Br'].values, B_Residual['Br'].values),
                   'Btheta':calculate_rms_error(B_Model_SVD['Btheta'].values, B_Residual['Btheta'].values),
                   'Bphi':calculate_rms_error(B_Model_SVD['Bphi'].values, B_Residual['Bphi'].values),
                   'Btotal':calculate_rms_error(B_Model_SVD['Btotal'].values, B_Residual['Btotal'].values)
                   }
        if RMS_df.empty:
            RMS_df = pd.DataFrame([new_row])
        else:
            RMS_df = pd.concat([RMS_df,pd.DataFrame([new_row])],ignore_index=True)
        print(f'Nmax = {Nmax} Model RMS Calculated')

    # Set the plotting style
    sns.set(style='whitegrid')

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 13), sharex=True)

    # Titles for subplots
    titles = ['Br', 'Btheta', 'Bphi', 'Btotal']

    # Plot each component in a separate subplot
    for i, title in enumerate(titles):
        ax = axes[i // 2, i % 2]  # Determine the position of the subplot
        sns.lineplot(data=RMS_df, x='Nmax', y=title, ax=ax, marker='o')

        ax.set_title(title+' RMS')
        ax.set_xlabel(f'Degree N')
        ax.set_ylabel('RMS value')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{path}/Model_RMS.jpg',dpi=400)
    plt.show()

def PLot_Bfield_Model(data,B_Residual,Nmax_List = [1,2,3],path = 'Spherical_Harmonic_Model/First50_Orbit_Model_External',Method='SVD'):

    Time_start = data.index.min()
    Time_end = data.index.max()

    for Nmax in Nmax_List:
        B_Model_SVD = calculate_Bfield(data, Nmax=Nmax, path=path, method=Method)

        os.makedirs(path+f'/InversionTest_Picture/{Nmax}',exist_ok=True)
        # Plot the magnetic field components and RMS errors
        plt.figure(figsize=(15, 10))

        # Plot Br component
        plt.subplot(6, 1, 1)
        component = 'Br'
        plt.plot(data.index, B_Residual[component], label=f'{component}_Residual', color='black')
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_Residual[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}',color='green')
        plt.title(f'Nmax={Nmax}'
                  f'\n{Time_start}-{Time_end}\n'
                  f'{component}')
        plt.ylabel(f'{component} (nT)')
        plt.legend()

        # Plot Btheta component
        plt.subplot(6, 1, 2)
        component = 'Bphi'
        plt.plot(data.index, B_Residual[component], label=f'{component}_Residual', color='black')
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_Residual[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
        plt.title(f'Nmax={Nmax}'
                  f'\n{Time_start}-{Time_end}\n'
                  f'{component}')
        plt.ylabel(f'{component} (nT)')
        plt.legend()

        # Plot Bphi component
        plt.subplot(6, 1, 3)
        component = 'Btheta'
        plt.plot(data.index, B_Residual[component], label=f'{component}_Residual', color='black')
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_Residual[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
        plt.title(f'Nmax={Nmax}'
                  f'\n{Time_start}-{Time_end}\n'
                  f'{component}')
        plt.ylabel(f'{component} (nT)')
        plt.legend()

        # Plot Bphi component
        plt.subplot(6, 1, 4)
        component = 'Btotal'
        plt.plot(data.index, B_Residual[component], label=f'{component}_Residual', color='black')
        RMS = calculate_rms_error(B_Model_SVD[component].values, B_Residual[component].values)
        plt.plot(data.index, B_Model_SVD[component], label=f'{component}_model_SVD RMS={RMS:.2f}', color='green')
        plt.title(f'Nmax={Nmax}'
                  f'\n{Time_start}-{Time_end}\n'
                  f'{component}')
        plt.ylabel(f'{component} (nT)')
        plt.legend()

        plt.subplot(6, 1, 5)
        component = 'r'
        plt.plot(data.index,data[component],label=f'{component}')
        plt.ylabel(f'{component} (Rj)')
        plt.xlabel('Time')
        plt.title(f'{component} ')
        plt.legend()


        plt.subplot(6, 1, 6)
        component = 'LocalTime'
        plt.plot(data.index, data[component], label=f'{component}')
        plt.ylabel(f'{component}')
        plt.xlabel('Time')
        plt.title(f'{component} ')
        plt.legend()


        # Adjust layout and show/save the figure
        plt.tight_layout()
        plt.savefig(path + f'/InversionTest_Picture/{Nmax}' + f'/Model_Bfield_{Time_start}.jpg', dpi=300)
        plt.close()
        # plt.show()
        print(f'Loop Ends Nmax = {Nmax}')
        print('-' * 50)


if __name__ == '__main__':

    def Model_Train():
        data = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_Data_1s_2h.csv')
        B_Residual = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_Residual_1s_2h.csv')

        # sample it 60s
        data = data.iloc[::60]
        B_Residual = B_Residual.iloc[::60]
        print('Hello')

        path = f'Spherical_Harmonic_Model/First50_Orbit_Model_External'

        Model_Simulation(data, B_Residual,NMIN=1,NMAX=5,SVD_rcond=1e-15,path=path)

    def Model_Test():
        # Test date
        year_doy_pj = {'2021': [[52, 32]]}

        # read the data
        data_test = read_data(year_doy_pj)
        Nmax_List  = [1,2,3,4,5]

        B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data_test)
        Model = 'jrm33'
        B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data_test, model=Model, degree=30)

        B_Residual = Juno_Mag_MakeData_Function.Caluclate_B_Residual(data_test, B_In=B_In, B_Ex=B_Ex)

        Plot_RMS_Nmax(data_test,B_Residual,Nmax_List=Nmax_List)
        PLot_Bfield_Model(data_test,B_Residual,Nmax_List=Nmax_List)

    Model_Test()
