import pandas as pd
import os
import numpy as np
from scipy.special import lpmn,factorial
import CoordinateTransform
import Juno_Mag_MakeData_Function
import seaborn as sns
import matplotlib.pyplot as plt
import re



def Model_Simulation(data, B_In_obs, Nmax_Internal=1,Nmax_External=1, SVD_On=True, SVD_rcond= 1e-15,path='Spherical_Harmonic_Model'):

    print(f'Start to train Model\nInternal Nmax = {Nmax_Internal}, External Nmax = {Nmax_External}')
    os.makedirs(path, exist_ok=True)

    data['theta'] = data['theta'] / 360 * 2 * np.pi
    data['phi'] = data['phi'] / 360 * 2 * np.pi

    # Total number of gnm and hnm coefficients
    num_coeffs = (Nmax_Internal + 2) * Nmax_Internal
    print(f'Total Number of Internal Field Model Coefficients = {num_coeffs}')
    num_coeffs = (Nmax_External + 2) * Nmax_External
    print(f'Total Number of External Field Model Coefficients = {num_coeffs}')

    # Initialize your observations vector B
    B = np.vstack((B_In_obs['Br'], B_In_obs['Btheta'], B_In_obs['Bphi'])).T.reshape(-1)
    # B_Double = np.hstack((B,B))
    # B_Double = np.concatenate([B, B])
    # Function to calculate the Schmidt semi-normalization factor

    # Populate the design matrix A

    print(f'The Shape of B Field is {B.shape}')

    # Calculate the Schmidt Matrix
    A_External = Schmidt_Matrix_External(data, Nmax_External)
    A_Internal = Schmidt_Matrix_Internal(data, Nmax_Internal)

    A_combined = np.hstack((A_Internal, A_External))
    print(f'Shape of A {A_combined.shape}')
    if SVD_On:

        U, sigma, VT = np.linalg.svd(A_combined, full_matrices=False)
        A_inv = np.linalg.pinv(A_combined, rcond=SVD_rcond)
        gnm_hnm_SVD = np.dot(A_inv, B)

        np.save(f'{path}/ExAndInternal_Inversion_SVD_coefficients_gnm_hnm_Nmax_{Nmax_Internal}_{Nmax_External}.npy', gnm_hnm_SVD)
        np.save(f'{path}/ExAndInternal_Inversion_SVD_coefficients_U_Nmax_{Nmax_Internal}_{Nmax_External}.npy', U)
        np.save(f'{path}/ExAndInternal_Inversion_SVD_coefficients_S_Nmax_{Nmax_Internal}_{Nmax_External}.npy', sigma)
        np.save(f'{path}/ExAndInternal_Inversion_SVD_coefficients_V_Nmax_{Nmax_Internal}_{Nmax_External}.npy', VT)

        print(f'The SVD Shape of the gnm_hnm is {gnm_hnm_SVD.shape}')
        print(f'The SVD Spape of U is {U.shape}, S is {sigma.shape}, V is {VT.shape}')
        print(f'SVD Nmax= IN {Nmax_Internal} EX {Nmax_External} finished')
        print('-'*50)


        print(f'Model Nmax = IN {Nmax_Internal} EX {Nmax_External} End')

        if (SVD_On) == False:
            print('No Model Trained!')
        print('=' * 50)

    data['theta'] = data['theta'] * 360 / (2*np.pi)
    data['phi'] = data['phi']  * 360 / (2*np.pi)

# Function to calculate the Schmidt semi-normalization factor
def schmidt_semi_normalization(n, m):
    return ((-1)**m)*np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))

def Schmidt_Matrix_Internal(data, Nmax):
    # Initialize the design matrix A
    num_coeffs = int((Nmax + 2) * Nmax)
    # print(f'Internal Schmidt Coefficient total numbers = {num_coeffs}\ngnm_num={(Nmax+3)*Nmax/2} hnm_num={(Nmax+3)*Nmax/2-Nmax}')
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
    # print(f'Internal SchmidtMatrix calculate success. Shape = {A.shape}\n'+'+'*50)

    return A

def Schmidt_Matrix_External(data, Nmax):
    # Initialize the design matrix A
    num_coeffs = int((Nmax + 2) * Nmax)
    # print(f'External Schmidt Coefficient total numbers = {num_coeffs}\ngnm_num={(Nmax+3)*Nmax/2} hnm_num={(Nmax+3)*Nmax/2-Nmax}')
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
    # print(f'External SchmidtMatrix calculate success. Shape = {A.shape}\n'+'+'*50)

    return A

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
                Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 2



            data_day = Data.loc[Time_start:Time_end]

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    # data.index = data['Time']
    return data
def read_gnm_hnm_data(method='SVD', Nmax_Internal=1,Nmax_External=1, path='Spherical_Harmonic_Model'):

    gnm_hnm_coeffi = np.load(f'{path}/ExAndInternal_Inversion_{method}_coefficients_gnm_hnm_Nmax_{Nmax_Internal}_{Nmax_External}.npy')

    return gnm_hnm_coeffi

def calculate_Bfield(data_input,path='Spherical_Harmonic_Model',Nmax_Internal=1,Nmax_External=1,method='SVD',Coordinate='Sys3'):

    # Set SS coordiante
    if Coordinate == 'SS':
        data = data_input[['Br_ss', 'Btheta_ss', 'Bphi_ss', 'r_ss', 'theta_ss', 'phi_ss']]
        data = data.rename(columns={'r_ss': 'r', 'theta_ss': 'theta', 'phi_ss': 'phi',
                                      'Br_ss': 'Br', 'Btheta_ss': 'Btheta', 'Bphi_ss': 'Bphi'})
    elif Coordinate=='Sys3':
        data = data_input

    data['theta'] = data['theta'] / 360 * 2 * np.pi
    data['phi'] = data['phi'] / 360 * 2 * np.pi

    SchmidtMatrix_External = Schmidt_Matrix_External(data, Nmax_External)
    SchmidtMatrix_Internal = Schmidt_Matrix_Internal(data, Nmax_Internal)
    SchmidtMatrix_Combined = np.hstack((SchmidtMatrix_Internal,SchmidtMatrix_External))

    gnm_hnm_coeffi = read_gnm_hnm_data(path=path,Nmax_Internal=Nmax_Internal,Nmax_External=Nmax_External,method=method)
    B_Model = np.dot(SchmidtMatrix_Combined,gnm_hnm_coeffi)

    B_Model = B_Model.reshape((int(len(B_Model)/3),3))
    # mid_point = len(B_Model) // 2
    # B_Model = B_Model[:mid_point]

    B_Model_df = pd.DataFrame(B_Model,columns=['Br','Btheta','Bphi'],index=data['Br'].index)


    B_Model_df['Btotal'] = np.sqrt(B_Model_df['Br']**2 + B_Model_df['Btheta']**2 + B_Model_df['Bphi']**2)


    print(f'B Field of Model Calculated \n Nmax= IN {Nmax_Internal} EX {Nmax_External}')

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

    if Coordinate == 'SS':
        B_Model_df = B_Model_df.rename(columns={'Br': 'Br_ss', 'Btheta': 'Btheta_ss', 'Bphi': 'Bphi_ss',
                                                  'Bx': 'Bx_ss', 'By': 'By_ss', 'Bz': 'Bz_ss'})


    return B_Model_df

def calculate_rms_error(B_pred, B_obs):
    return np.sqrt(np.mean((B_pred - B_obs)**2))

def Plot_RMS_Nmax(data,B_Residual,Nmax_List_In = [1,2,3],Nmax_List_Ex = [1,2,3],Coordinate='Sys3',path = 'Spherical_Harmonic_Model/First50_Orbit_Model_External',Method='SVD'):

    RMS_df = pd.DataFrame()

    # Titles for subplots
    if Coordinate == 'Sys3':
        titles = ['Br', 'Btheta', 'Bphi', 'Btotal']
    if Coordinate == 'SS':
        titles = ['Br_ss', 'Btheta_ss', 'Bphi_ss', 'Btotal']

    for Nmax_Internal in Nmax_List_In:
        for Nmax_External in Nmax_List_Ex:
            try:
                B_Model_SVD = calculate_Bfield(data, Nmax_Internal=Nmax_Internal, Nmax_External=Nmax_External, path=path,
                                               method=Method,Coordinate=Coordinate)

            except:
                print(f'No Model In{Nmax_Internal} Ex{Nmax_External} Found!\npath={path}')


            new_row = {'Nmax':Nmax_Internal+Nmax_External,
                       'Nmax_Internal':Nmax_Internal,
                       'Nmax_External':Nmax_External,
                       }

            for comp in titles:
                new_row[comp] = calculate_rms_error(B_Model_SVD[comp].values, B_Residual[comp].values)

            if RMS_df.empty:
                RMS_df = pd.DataFrame([new_row])
            else:
                RMS_df = pd.concat([RMS_df,pd.DataFrame([new_row])],ignore_index=True)
            print(f'Nmax = IN {Nmax_Internal} EX{Nmax_External} Model RMS Calculated')

    # Set the plotting style
    sns.set(style='whitegrid')

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 13), sharex=True)



    # Plot each component in a separate subplot
    for i, title in enumerate(titles):
        ax = axes[i // 2, i % 2]  # Determine the position of the subplot
        sns.scatterplot(data=RMS_df, x='Nmax', y=title, ax=ax, marker='o', s=100)

        # sns.lineplot(data=RMS_df, x='Nmax', y=title, ax=ax, marker='', alpha=0.3)

        # Adding annotations for each point
        for line in range(0, RMS_df.shape[0]):
            ax.text(RMS_df.Nmax[line] + 0.1, RMS_df[title][line],
                    f'({RMS_df.Nmax_Internal[line]},{RMS_df.Nmax_External[line]})',
                    horizontalalignment='left', size='small', color='black')

        ax.set_title(title+' RMS'+f' Coordinate:{Coordinate}')
        ax.set_xlabel(f'Degree N (Internal+External)')
        ax.set_ylabel('RMS value')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{path}/Model_RMS.jpg',dpi=400)
    plt.show()


def plot_component(ax, data, component, label, color):
    ax.plot(data.index, data[component], label=f'{label} {component}', color=color)
    ax.set_ylabel(f'{component} (nT)')
    ax.legend()


def PLot_Bfield_Model(data, B_Residual, Nmax_List_In=[1, 2, 3], Nmax_List_Ex=[1, 2, 3], Coordinate='Sys3',
                      path='Spherical_Harmonic_Model/First50_Orbit_Model_External', Method='SVD'):
    Time_start = data.index.min()
    Time_end = data.index.max()

    if Coordinate == 'Sys3':
        components = ['Br', 'Btheta', 'Bphi', 'Btotal','r','LocalTime']
    if Coordinate == 'SS':
        components = ['Br_ss', 'Btheta_ss', 'Bphi_ss', 'Btotal','r_ss','LocalTime']

    for Nmax_Internal in Nmax_List_In:
        for Nmax_External in Nmax_List_Ex:

            try:
                B_Model_SVD = calculate_Bfield(data, Nmax_Internal=Nmax_Internal, Nmax_External=Nmax_External,
                                               path=path, method=Method,Coordinate=Coordinate)

            except Exception as e:
                print(f'Error for IN{Nmax_Internal} EX{Nmax_External}: {e}\nPath={path}')
                continue

            dir_path = os.path.join(path, f'InversionTest_Picture/IN{Nmax_Internal}_Ex{Nmax_External}')
            os.makedirs(dir_path, exist_ok=True)

            plt.figure(figsize=(15, 10))
            for i, component in enumerate(components):
                ax = plt.subplot(6, 1, i + 1)
                if component in B_Residual and component in B_Model_SVD:
                    plot_component(ax, B_Residual, component, 'Residual', 'black')
                    RMS = calculate_rms_error(B_Model_SVD[component].values, B_Residual[component].values)
                    plot_component(ax, B_Model_SVD, component, f'Model SVD RMS={RMS:.2f}', 'green')
                elif component in data:
                    plot_component(ax, data, component, component, 'blue')
                ax.set_title(f'{component} from {Time_start} to {Time_end}')
                ax.set_xlabel('Time')

            plt.tight_layout()
            plt.savefig(os.path.join(dir_path, f'Model_Bfield_{Time_start}.jpg'), dpi=300)
            plt.close()

            print(f'Loop Ends Nmax = IN{Nmax_Internal} EX{Nmax_External}')
            print('-' * 50)


def plot_rms_data_Orbits(RMS_df, titles,path,Nmax_Internal, Nmax_External):
    x_axes = ['PJ', 'Longitude', 'LocalTime']  # The different x-axes for plotting

    fig, axes = plt.subplots(nrows=len(titles), ncols=3, figsize=(18, 10 * len(titles)))
    # Adjust the size and layout dynamically based on the number of components
    plt.subplots_adjust(hspace=0.6, wspace=0.4)  # Adjust space between plots

    bar_width = 0.35  # Width of the bars in the bar plot

    fig.suptitle(f'Magnetic Field Component RMS Errors - Model Degree Internal: {Nmax_Internal}, External: {Nmax_External}', fontsize=16)

    for i, component in enumerate(titles):
        for j, x_axis in enumerate(x_axes):
            ax = axes[i][j]
            # Sort the DataFrame by the x_axis if it's Longitude_ss or LocalTime
            if x_axis in ['Longitude', 'LocalTime']:
                sorted_df = RMS_df.sort_values(by=x_axis)
                indices = np.arange(len(sorted_df[x_axis]))
            else:
                sorted_df = RMS_df
                indices = np.arange(len(sorted_df[x_axis]))

            ax.bar(indices, sorted_df[f'{component}_JRM'], width=bar_width, label=f'{component} JRM', color='blue')
            ax.bar(indices + bar_width, sorted_df[f'{component}_Model'], width=bar_width, label=f'{component} Model',
                   color='red', alpha=0.8)

            ax.set_title(f'{component} RMS vs. {x_axis}')
            ax.set_xlabel(x_axis)
            ax.set_ylabel('RMS Error')

            # Formatting and setting ticks and labels
            if x_axis == 'Longitude':
                formatted_labels = [f"{value:.0f}" for value in sorted_df[x_axis]]
                # ax.set_xlim([0, 360])  # Set x-axis limits for longitude
                plt.xticks(np.linspace(0, 360, num=12))
            elif x_axis == 'LocalTime':
                formatted_labels = [f"{value:.0f}" for value in sorted_df[x_axis]]
                # ax.set_xlim([0, 24])  # Set x-axis limits for local time
                plt.xticks(np.linspace(0, 24, num=12))
            else:
                formatted_labels = sorted_df[x_axis]

            ax.set_xticks(indices[::max(1, len(indices) // 10)] + bar_width / 2)
            ax.set_xticklabels(formatted_labels[::max(1, len(indices) // 10)],
                               rotation=45)  # Rotate labels for better fit
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(path, f'Model_RMS_50Orbits.jpg'), dpi=300)
    plt.show()
    plt.close()

def Plot_RMS_Orbits(Nmax_Internal=1,Nmax_External=1,Coordinate='SS',path = 'Spherical_Harmonic_Model/First50_Orbit_Model_External',Method='SVD'):

    year_doy_pj = {'2016': [[240, 1], [346, 3]],
                   '2017': [[33, 4], [86, 5], [139, 6], [191, 7], [244, 8], [297, 9], [350, 10]],
                   '2018': [[38, 11], [91, 12], [144, 13], [197, 14], [249, 15], [302, 16], [355, 17]],
                   '2019': [[43, 18], [96, 19], [149, 20], [201, 21], [254, 22], [307, 23], [360, 24]],
                   '2020': [[48, 25], [101, 26], [154, 27], [207, 28], [259, 29], [312, 30], [365, 31]],
                   '2021': [[52, 32], [105, 33], [159, 34], [202, 35], [245, 36], [289, 37], [333, 38]],
                   '2022': [[12, 39], [55, 40], [99, 41], [142, 42], [186, 43], [229, 44], [272, 45], [310, 46],
                            [348, 47]],
                   '2023': [[22, 48], [60, 49], [98, 50]]}

    # year_doy_pj = {'2016': [[240, 1], [346, 3]],
    #                '2017': [[33, 4], [86, 5], [139, 6], [191, 7], [244, 8], [297, 9], [350, 10]]}
    RMS_df = pd.DataFrame()


    if Coordinate == 'Sys3':
        titles = ['Br', 'Btheta', 'Bphi', 'Btotal']
    if Coordinate == 'SS':
        titles = ['Br_ss', 'Btheta_ss', 'Bphi_ss', 'Btotal']

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            # date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

            # read data
            Data = Juno_Mag_MakeData_Function.Read_Data(year_doy,freq=1)

            # 24 hours data
            Time_start = Data.index.min()
            Time_end = Data.index.max()

            data_day = Data.loc[Time_start:Time_end]
            data_day = data_day[::60]

            B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data_day)
            B_Ex_day = CoordinateTransform.SysIIItoSS_Bfield(data_day,B_Ex_day)
            Model = 'jrm33'
            B_In_day = Juno_Mag_MakeData_Function.MagneticField_Internal(data_day, model=Model, degree=30)
            B_In_day = CoordinateTransform.SysIIItoSS_Bfield(data_day,B_In_day)


            # The Residual Model
            B_Model_day = calculate_Bfield(data_day, Nmax_Internal=Nmax_Internal, Nmax_External=Nmax_External,
                                               path=path, method=Method,Coordinate=Coordinate)
            B_JRM = B_In_day + B_Ex_day
            B_Model = B_In_day + B_Ex_day + B_Model_day

            new_row = {'Nmax': Nmax_Internal + Nmax_External,
                       'Nmax_Internal': Nmax_Internal,
                       'Nmax_External': Nmax_External,
                       'PJ':pj,
                       'Longitude_ss': data_day['Longitude_ss'].loc[data_day['r'].idxmin()],
                       'Longitude': data_day['Longitude'].loc[data_day['r'].idxmin()],
                       'LocalTime': data_day['LocalTime'].loc[data_day['r'].idxmin()]
                       }

            for comp in titles:
                new_row[comp+'_JRM'] = calculate_rms_error(B_JRM[comp].values, data_day[comp].values)
                new_row[comp+'_Model'] = calculate_rms_error(B_Model[comp].values, data_day[comp].values)

            if RMS_df.empty:
                RMS_df = pd.DataFrame([new_row])
            else:
                RMS_df = pd.concat([RMS_df, pd.DataFrame([new_row])], ignore_index=True)

    plot_rms_data_Orbits(RMS_df,titles,path,Nmax_Internal, Nmax_External)



def save_Gnm_Hnm_toCsv(path):
    # Specify the directory containing your .npy files
    directory = path

    # List to hold each DataFrame
    data_frames = []

    pattern = r'^ExAndInternal_Inversion_SVD_coefficients_gnm_hnm_Nmax_(\d+)_(\d+)\.npy$'

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Load the .npy file
            array = np.load(file_path)

            flat_array = array.flatten()

            match = re.match(pattern, filename)
            special_num1, special_num2 = int(match.group(1)), int(match.group(2))

            # Convert the array to a DataFrame
            df = pd.DataFrame([flat_array])

            # Add the special numbers as new columns at the beginning
            df.insert(0, 'Internal_Nmax', special_num1)
            df.insert(1, 'External_Nmax', special_num2)

            # Append the DataFrame to the list
            data_frames.append(df)

    # Concatenate all DataFrames into one
    final_df = pd.concat(data_frames, ignore_index=True)

    final_df.sort_values(by=['Internal_Nmax', 'External_Nmax'], inplace=True)
    # Save the final DataFrame to a single CSV file
    final_df.to_csv('Gnm_Hnm_InAndEx.csv', index=False)

    print('All .npy files have been converted and combined into one CSV file.')

if __name__ == '__main__':

    def Model_Train(path = f'Spherical_Harmonic_Model/First50_Orbit_Model_ExAndInternal'):
        data = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_Data_60s_24h.csv')
        B_Residual = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_Residual_60s_24h.csv')

        # sample it 60s
        data = data.iloc[::60]
        print(data.keys())
        # B_Ex = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_B_Ex_1s_2h.csv')
        # B_Ex = B_Ex.iloc[::60]
        # B_In_obs = Spherical_Harmonic_InversionModel_Functions.B_In_obs_Calculate(data, B_Ex)

        B_Residual = B_Residual.iloc[::60]
        print(B_Residual.keys())
        B_Residual_SS = B_Residual[['Br_ss','Btheta_ss','Bphi_ss']]
        B_Residual_SS = B_Residual_SS.rename(columns={'Br_ss':'Br','Btheta_ss':'Btheta','Bphi_ss':'Bphi'})
        data_ss = data[['r_ss','theta_ss','phi_ss']]
        data_ss = data_ss.rename(columns={'r_ss':'r','theta_ss':'theta','phi_ss':'phi'})
        print(data_ss.describe())

        # Model_Simulation(data, B_Residual,NMIN=1,NMAX=6,SVD_rcond=1e-15,path=path)
        Nmax_List_Internal = list(range(1,11))
        Nmax_List_External = list(range(1,4))
        for Nmax_Internal in Nmax_List_Internal:
            for Nmax_External in Nmax_List_External:
                Model_Simulation(data_ss, B_Residual_SS,Nmax_Internal=Nmax_Internal,Nmax_External=Nmax_External,SVD_rcond=1e-15,path=path)

    def Model_Test(path=f'Spherical_Harmonic_Model/First50_Orbit_Model_ExAndInternal'):
        # Test date
        year_doy_pj = {'2021': [[52, 32]]}

        # read the data
        data_test = read_data(year_doy_pj,time_period=2)
        # Nmax_List  = list(range(1,6))
        print(data_test.keys())
        print(data_test['Longitude'].describe())

        B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data_test)
        B_Ex = CoordinateTransform.SysIIItoSS_Bfield(data_test,B_Ex)

        Model = 'jrm33'
        B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data_test, model=Model, degree=30)
        B_In = CoordinateTransform.SysIIItoSS_Bfield(data_test,B_In)

        B_Residual = Juno_Mag_MakeData_Function.Caluclate_B_Residual(data_test, B_In=B_In, B_Ex=B_Ex,Coor_SS=True)

        Nmax_List_Internal = list(range(1, 11))
        Nmax_List_External = list(range(1, 3))
        # Plot_RMS_Nmax(data_test,B_Residual,Nmax_List_In=Nmax_List_Internal,Nmax_List_Ex=Nmax_List_External,path=path,Coordinate='SS')
        PLot_Bfield_Model(data_test,B_Residual,Nmax_List_In=Nmax_List_Internal,Nmax_List_Ex=Nmax_List_External,path=path,Coordinate='SS')


    path = f'Spherical_Harmonic_Model/First50_Orbit_Model_ExAndInternal_24h'
    # Model_Train(path)
    # Model_Test(path)
    # save_Gnm_Hnm_toCsv(path='Spherical_Harmonic_Model/First50_Orbit_Model_ExAndInternal')
    Plot_RMS_Orbits(Nmax_Internal=1,Nmax_External=1,path=path)