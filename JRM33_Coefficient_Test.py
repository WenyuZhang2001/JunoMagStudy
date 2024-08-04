import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import CoordinateTransform
import Juno_Mag_MakeData_Function
from scipy.special import lpmn,factorial
import joblib
import os
import Spherical_Harmonic_InversionModel_Functions
import seaborn as sns


def calculate_Bfield(data,Nmax=10,Rc=1.0):
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

    SchmidtMatrix = Spherical_Harmonic_InversionModel_Functions.Schmidt_Matrix(data,Nmax)
    # ridge_model = joblib.load(f'{path}/ridge_model_Nmax{Nmax}_Alpha{Ridge_alpha}.joblib')
    # B_Model = ridge_model.predict(SchmidtMatrix)
    gnm_hnm_coeffi = np.load('Result_data/JRM33_Coefficient.npy')
    Spherical_Harmonic_InversionModel_Functions.ParameterScale(gnm_hnm_coeffi,Nmax=Nmax,Rc=Rc)
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

if __name__ == '__main__':
    year_doy = {'2023': [136]}
    # read data
    Data = Juno_Mag_MakeData_Function.Read_Data(year_doy, freq=1)

    # 2h data
    PeriJovian_time = Data['r'].idxmin()
    # 2 hour window data
    Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
    Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 3

    data_day = Data.loc[Time_start:Time_end]
    data_day = data_day[::60]

    B_my_calculation = calculate_Bfield(data=data_day,Nmax=30,Rc=1)
    B_Jrm = Juno_Mag_MakeData_Function.MagneticField_Internal(data_day,degree=30)

    B_residual = B_my_calculation-B_Jrm
    print(B_residual.describe())
    # Set the plotting style
    sns.set(style='whitegrid')

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8), sharex=True)

    # Titles for subplots
    titles = ['Br', 'Btheta', 'Bphi', 'Btotal']

    # Plot each component in a separate subplot
    for i, title in enumerate(titles):
        ax = axes[i // 2, i % 2]  # Determine the position of the subplot
        sns.lineplot(data=B_my_calculation, x=B_my_calculation.index, y=title, ax=ax, marker='o',label='B_my_calculation')
        sns.lineplot(data=B_Jrm, x=B_Jrm.index, y=title, ax=ax, marker='*',label='JRM33_Code')

        ax.set_title(title)

        ax.set_ylabel(title)

    # Adjust layout
    plt.tight_layout()
    plt.show()
