import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import Juno_Mag_MakeData_Function
import Spherical_Harmonic_InversionModel_Functions




def Check_SVD_SV(path,Nmax=20,Nmin=1,Method='SVD'):

    os.makedirs(path + f'/InversionTest_Picture', exist_ok=True)

    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.title('Each Nmax SVD Model Eigenvalue')
    for N in range(Nmin,Nmax):
        try:
            S = np.load(f'{path}/Inversion_{Method}_coefficients_S_Nmax{N}.npy')
        except:
            continue
        plt.plot(S,marker='o',linestyle='-',alpha=0.5)
        plt.yscale('log')

        plt.text(len(S),S[-1],f'Nmax={N}',rotation=45)

    plt.ylabel('Eigenvalue log')
    plt.xlabel('Index')
    plt.grid()


    plt.subplot(2,1,2)
    plt.title('Each Nmax SVD Model Resolution Matrix Trace')
    for N in range(Nmin,Nmax):
        try:
            V = np.load(f'{path}/Inversion_{Method}_coefficients_V_Nmax{N}.npy')
        except:
            continue
        Trace = np.trace(np.dot(V,V.T))
        plt.plot(N,Trace,marker='o',linestyle='-')

    plt.ylabel('Trace')
    plt.xlabel('Nmax')
    plt.grid()
    plt.savefig(path+f'/InversionTest_Picture/{Method}_Model_Eg_Trace.jpg',dpi=300)
    plt.show()


def Check_SVD_Trace(path,Nmax=20,Nmin=1,Method='SVD'):
    plt.figure()
    for N in range(Nmin,Nmax):
        V = np.load(f'{path}/Inversion_{Method}_coefficients_V_Nmax{N}.npy')
        R = np.dot(V,V.T)
        plt.imshow(R, cmap='viridis', interpolation='nearest',vmin=0.5,vmax=1)
        plt.colorbar()
        plt.title('Resolution Matrix R')
        plt.show()

def Check_Magnetic_Spectrum(path,Nmax=20,ParameterScaleOn=False,Rc=1,Method='SVD',Show_Pic=True):
    os.makedirs(path + f'/InversionTest_Picture/', exist_ok=True)
    gnm_hnm_coeffi = Spherical_Harmonic_InversionModel_Functions.read_gnm_hnm_data(method=Method, Nmax=Nmax,path=path)
    if ParameterScaleOn:
        Spherical_Harmonic_InversionModel_Functions.ParameterScale(gnm_hnm_coeffi,Nmax=Nmax)
    Rn = np.zeros(Nmax)
    for n in range(1, Nmax + 1):
        for m in range(n + 1):
            # gnm index
            # (n-1+3)*(n-1)/2 + m
            gnm_index = int((n + 2) * (n - 1) / 2 + m)
            # hnm index
            # (n-1+3)*(n-1)/2 - (n-1) + m-1 + gnm_num (= (n+3)*n/2
            hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)

            Rn[n-1] += (n+1) * ( gnm_hnm_coeffi[gnm_index]**2 + gnm_hnm_coeffi[hnm_index]**2)

    plt.figure()
    plt.plot(range(1,Nmax+1),Rn,marker='o',linestyle='-')
    plt.ylabel(r'Rn $(nT)^{2}$')
    plt.yscale('log')
    plt.xlabel('Degree n')
    plt.title('Magnetic Spectrum of the SVD gnm and hnm')
    plt.grid()
    plt.savefig(path + f'/InversionTest_Picture/{Method}_Model_MagneticSpectrum.jpg', dpi=300)
    if Show_Pic:
        plt.show()
    plt.close()
def calculate_rms_error(B_pred, B_obs):
    return np.sqrt(np.mean((B_pred - B_obs) ** 2))

def Model_RMS_Rc(Rc,Lambda,path,Nmax=30,Method='Regularized_SVD'):
    year_doy = {'2021': [52]}
    # read data
    Data = Juno_Mag_MakeData_Function.Read_Data(year_doy, freq=1)

    # 2h data
    PeriJovian_time = Data['r'].idxmin()
    # 2 hour window data
    Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
    Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 2

    data_day = Data.loc[Time_start:Time_end]

    B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data_day)


    B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data_day, B_Ex_day)

    path_lambda = path + f'{Rc}_{Lambda:.2e}'
    B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data_day, Nmax=Nmax, path=path_lambda,
                                                                               method=Method)
    RMS = {
               'Br': calculate_rms_error(B_Model_SVD['Br'].values, B_In_obs['Br'].values),
               'Btheta': calculate_rms_error(B_Model_SVD['Btheta'].values, B_In_obs['Btheta'].values),
               'Bphi': calculate_rms_error(B_Model_SVD['Bphi'].values, B_In_obs['Bphi'].values),
               'Btotal': calculate_rms_error(B_Model_SVD['Btotal'].values, B_In_obs['Btotal'].values)
               }

    return RMS


def Plot_MagneticSpectrum_Rc(Rc_Lambda_Dic={'1.0':[1.0]},Nmax=30,path = 'Spherical_Harmonic_Model/First50_Orbit_Model_Regularization_',
                             Method='Regularized_SVD',Regularize_On=True,SaveSingleModelPlot=False,JRM_Rc=1.0,Savefig=True,Check_RMS=True,RMS_Threshhold=150):

    Rn_df = pd.DataFrame(columns=['Rc','Lambda','Rn'])

    for Rc in Rc_Lambda_Dic.keys():
        for Lambda in Rc_Lambda_Dic[Rc]:

            path_Rc = path+f'{Rc}_{Lambda:.2e}'
            print(path_Rc)
            # Draw each models Magnetic spectrum and save to file
            if SaveSingleModelPlot:
                Check_Magnetic_Spectrum(path=path_Rc,Nmax=Nmax,ParameterScaleOn=False,Method=Method,Show_Pic=False)
            if Check_RMS:
                RMS = Model_RMS_Rc(Rc,Lambda,path,Nmax=Nmax,Method=Method)
                if any(value > RMS_Threshhold for value in RMS.values()):
                    print(f'RMS Value Over 150 Rc{Rc} Lambda{Lambda}')
                    print(RMS)
                    continue
            # calculate the Rn
            gnm_hnm_coeffi = Spherical_Harmonic_InversionModel_Functions.read_gnm_hnm_data(method=Method, Nmax=Nmax,
                                                                                           path=path_Rc)
            Rn = np.zeros(Nmax)
            for n in range(1, Nmax + 1):
                for m in range(n + 1):
                    # gnm index
                    # (n-1+3)*(n-1)/2 + m
                    gnm_index = int((n + 2) * (n - 1) / 2 + m)
                    # hnm index
                    # (n-1+3)*(n-1)/2 - (n-1) + m-1 + gnm_num (= (n+3)*n/2
                    hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)

                    if Regularize_On:
                        Rn[n - 1] += (n + 1) * (gnm_hnm_coeffi[gnm_index] ** 2 + gnm_hnm_coeffi[hnm_index] ** 2)*(1/float(Rc))**(2*n+4)
                    else:
                        Rn[n - 1] += (n + 1) * (gnm_hnm_coeffi[gnm_index] ** 2 + gnm_hnm_coeffi[hnm_index] ** 2)
            new_row = {'Rc':Rc,
                       'Lambda':Lambda,
                       'Rn':Rn
                       }
            if Rn_df.empty:
                Rn_df = pd.DataFrame([new_row])
            else:
                Rn_df = pd.concat([Rn_df,pd.DataFrame([new_row])],ignore_index=True)

    # Calculate JRM33 Magnetic Spectrum
    JRM_gnm_hnm = np.load('Result_data/JRM33_Coefficient.npy')
    Rn_JRM = np.zeros(Nmax)

    for n in range(1, Nmax + 1):
        for m in range(n + 1):
            # gnm index
            # (n-1+3)*(n-1)/2 + m
            gnm_index = int((n + 2) * (n - 1) / 2 + m)
            # hnm index
            # (n-1+3)*(n-1)/2 - (n-1) + m-1 + gnm_num (= (n+3)*n/2
            hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)

            if Regularize_On:
                Rn_JRM[n - 1] += (n + 1) * (JRM_gnm_hnm[gnm_index] ** 2 + JRM_gnm_hnm[hnm_index] ** 2) * (
                            1 / float(JRM_Rc)) ** (2 * n + 4)
            else:
                Rn_JRM[n - 1] += (n + 1) * (JRM_gnm_hnm[gnm_index] ** 2 + JRM_gnm_hnm[hnm_index] ** 2)

    # Plotting
    fig, ax = plt.subplots(figsize=(12,8))

    # Define colors for each unique Rc
    unique_rcs = Rn_df['Rc'].unique()
    color_map = {rc: plt.cm.jet(i / len(unique_rcs)) for i, rc in enumerate(unique_rcs)}

    # Define markers for each Lambda if needed
    markers = ['o', 's', 'v','^','>','<','1','2','3','4', 'x','*','P','p','+','h','H','X','D','d','|','_']

    # Define markers for each unique Lambda
    unique_lambdas = Rn_df['Lambda'].unique()
    Lambda_number = len(Rc_Lambda_Dic[next(iter(Rc_Lambda_Dic))])
    marker_map = {lambda_: ['o', 's', 'v','^','>','<','1','2','3','4', 'x','*','P','p','+','h','H','X','D','d','|','_'][i % Lambda_number] for i, lambda_ in enumerate(unique_lambdas)}

    for (idx, row) in Rn_df.iterrows():
        rc, lambda_, rn = row['Rc'], row['Lambda'], row['Rn']
        ax.plot(np.arange(1,len(rn)+1), rn, label=f'Rc={rc}, Lambda={lambda_}' if idx in [0, 1] else "",
                marker=marker_map[lambda_], color=color_map[rc], linestyle='-')

    # Creating custom legends
    # Legend for Rc
    rc_legend = [plt.Line2D([0], [0], color=color_map[rc], marker='o', linestyle='', label=f'Rc={rc}')
                 for rc in unique_rcs]
    # Legend for Lambda
    lambda_legend = [
        plt.Line2D([0], [0], color='black', marker=marker_map[lambda_], linestyle='', label=f'Lambda={lambda_}')
        for lambda_ in unique_lambdas]

    ax.plot(np.arange(1,Nmax+1),Rn_JRM,label='JRM33',color='lightpink',marker='o')
    # Add legends to the plot
    legend1 = ax.legend(handles=rc_legend, title="Rc Values", loc='center left',fontsize=6)
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=lambda_legend, title="Lambda Values", loc='lower left',fontsize=5.5)
    ax.add_artist(legend2)
    legend3 = ax.legend(handles=[plt.Line2D([0], [0], color='lightpink', marker='o', linestyle='-', label=f'JRM33 Rc={JRM_Rc}')],
                        title="Reference Model", loc='upper right', fontsize=6)
    ax.add_artist(legend3)

    ax.set_xlabel('Degree n')
    ax.set_ylabel(r'Rn $(nT)^{2}$')
    ax.set_title(f'Magnetic Spectrum Rn (RMS < {RMS_Threshhold})')
    ax.set_yscale('log')
    # ax.legend()
    plt.tight_layout()

    if Savefig==True:
        plt.savefig(f'Spherical_Harmonic_Model/Model_MagneticSpectrum_Rn_{Regularize_On}.jpg',dpi=400)
    plt.show()


if __name__ == '__main__':


    # path = 'Spherical_Harmonic_Model/First50_Orbit_Model_RegularizationTest'
    # Check_Magnetic_Spectrum(path,Nmax=40,ParameterScaleOn=True,Rc=1,Method='Regularized_SVD')
    # Check_SVD_SV(path=path,Nmax=41,Method='Regularized_SVD')
    # path = 'Spherical_Harmonic_Model/First50_Orbit_Model'
    # Check_Magnetic_Spectrum(path,Nmax=41,ParameterScaleOn=True,Rc=1,Method='SVD')

    Rc_Lambda_Dic = {
        '1.0': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
        '0.9': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
        '0.88': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                 1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
        '0.85': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                 1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
        '0.92': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                 1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
        '0.8': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
                1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
        '0.7': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13,
                1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20],

        }


    # Rc_Lambda_Dic = {
    #
    #
    #     '0.85': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15,
    #              1e-16, 1e-17, 1e-18, 1e-19, 1e-20],
    2#r
    #
    # }


    Plot_MagneticSpectrum_Rc(Rc_Lambda_Dic,SaveSingleModelPlot=False,Regularize_On=False,Savefig=True,Check_RMS=True,RMS_Threshhold=500)

