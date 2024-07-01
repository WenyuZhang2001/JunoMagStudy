# Example Python script to read model in hdf5 formatted file

# Model types:
# 00: Main Field Model
# 01: Main Field + Zonal Flux Velocity Model

# Model type 00 and 01:
# array g contains the magnetic field coefficients in the standard ordering 

# Model type 01:
# array v contains the zonal flux velocity coefficents (t1, t2, t3, ...)

# All coefficients are for Schmidt normalized associated Legendre polynomials.

import numpy as np
import h5py
from matplotlib import pyplot as plt
from scipy.special import lpmn,factorial
import pandas as pd
import CoordinateTransform
import Juno_Mag_MakeData_Function


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

def calculate_Bfield(data,path='BaselineModel/baseline_model.hdf5',Nmax=32):
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

    # Read coefficients
    with h5py.File(path, 'r') as hm:
        model_type = hm.attrs['model_type']
        itype = int(model_type[:2])
        B_dset = hm['B_field_coeffs']
        lmax = B_dset.attrs['lmax']
        g = np.array(B_dset)
        if itype == 1:
            v_dset = hm['zfv_coeffs']
            v = np.array(v_dset)
            lmaxv = v_dset.attrs['lmax']

    gnm_hnm_coeffi = g

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

def read_data(year_doy_pj,time_period=24):
    data = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year: [doy[0]]}
            date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

            # read data
            Data = Juno_Mag_MakeData_Function.Read_Data(year_doy, freq=1)
            Data = Data.iloc[::60]
            Data['PJ'] = pj

            if time_period==24:
                # 24 hours data
                Time_start = date_list['Time'].iloc[0]
                Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 24
            elif time_period == 2:
                PeriJovian_time = Data['r'].idxmin()
                Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
                Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 3



            data_day = Data.loc[Time_start:Time_end]

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    # data.index = data['Time']
    return data


if __name__ == '__main__':
    year_doy_pj = {'2021': [[52, 32]]}
    data_test = read_data(year_doy_pj, time_period=2)


    Baseline_df = calculate_Bfield(data_test)
    Baseline_df = CoordinateTransform.SysIIItoSS_Bfield(data_test, Baseline_df)

    B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data_test, model='jrm33', degree=30)
    B_In = CoordinateTransform.SysIIItoSS_Bfield(data_test, B_In)




