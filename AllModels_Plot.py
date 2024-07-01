import pandas as pd
import os
import numpy as np
from scipy.special import lpmn,factorial

import BaselineModel
import CoordinateTransform
import Data_Date
import Juno_Mag_MakeData_Function
import seaborn as sns
import matplotlib.pyplot as plt
import re
import Spherical_Harmonic_InversionModel_Functions
import Spherical_Harmonic_Inversion_ExAndInternal
from LSTM_B_Residual_Test import LSTM_Simulation


def calculate_rms_error(B_pred, B_obs):
    return np.sqrt(np.mean((B_pred - B_obs)**2))

def plot_rms_data_Orbits(RMS_df, titles,path,Nmax_Internal, Nmax_External,Model_Numbers=3):
    x_axes = ['PJ', 'Longitude', 'LocalTime']  # The different x-axes for plotting

    fig, axes = plt.subplots(nrows=len(titles), ncols=3, figsize=(18, 10 * len(titles)))
    # Adjust the size and layout dynamically based on the number of components
    plt.subplots_adjust(hspace=0.6, wspace=0.4)  # Adjust space between plots

    bar_width = 1.0/Model_Numbers  # Width of the bars in the bar plot

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

            ax.bar(indices, sorted_df[f'{component}_JRM'], width=bar_width, label=f'{component} JRM', color='royalblue')
            ax.bar(indices + bar_width, sorted_df[f'{component}_SVD'], width=bar_width,
                   label=f'{component} Residual SVD Nmax({Nmax_Internal},{Nmax_External})',color='firebrick')
            ax.bar(indices+ bar_width*2, sorted_df[f'{component}_LSTM'], width=bar_width, label=f'{component} LSTM', color='green')

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

            ax.set_xticks(indices[::max(1, len(indices) // 10)] + bar_width /Model_Numbers)
            ax.set_xticklabels(formatted_labels[::max(1, len(indices) // 10)],
                               rotation=45)  # Rotate labels for better fit
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(path, f'Model_RMS_50Orbits.jpg'), dpi=300)
    plt.show()
    plt.close()


def Plot_RMS_Orbits(degree=30,Nmax_Internal=1,Nmax_External=1,Coordinate='SS',Plot_Save_path='Result_pic',ExAndIn_path = 'Spherical_Harmonic_Model/First50_Orbit_Model_External',Method='SVD',
                    LSTM_path='LSTM/magnetic_field_model.h5'):

    year_doy_pj = {'2016': [[240, 1], [346, 3]],
                   '2017': [[33, 4], [86, 5], [139, 6], [191, 7], [244, 8], [297, 9], [350, 10]],
                   '2018': [[38, 11], [91, 12], [144, 13], [197, 14], [249, 15], [302, 16], [355, 17]],
                   '2019': [[43, 18], [96, 19], [149, 20], [201, 21], [254, 22], [307, 23], [360, 24]],
                   '2020': [[48, 25], [101, 26], [154, 27], [207, 28], [259, 29], [312, 30], [365, 31]],
                   '2021': [[52, 32], [105, 33], [159, 34], [202, 35], [245, 36], [289, 37], [333, 38]],
                   '2022': [[12, 39], [55, 40], [99, 41], [142, 42], [186, 43], [229, 44], [272, 45], [310, 46],
                            [348, 47]],
                   '2023': [[22, 48], [60, 49], [98, 50]]}
    # #
    # year_doy_pj = {'2023':[[98,50]]}
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

            # 2h data
            PeriJovian_time = Data['r'].idxmin()
            # 2 hour window data
            Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
            Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 3

            data_day = Data.loc[Time_start:Time_end]
            # data_day = data_day[::60]

            B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data_day)
            B_Ex_day = CoordinateTransform.SysIIItoSS_Bfield(data_day,B_Ex_day)
            Model = 'jrm33'
            B_In_day = Juno_Mag_MakeData_Function.MagneticField_Internal(data_day, model=Model, degree=degree)
            B_In_day = CoordinateTransform.SysIIItoSS_Bfield(data_day,B_In_day)


            # The Residual Model
            B_SVD_day = Spherical_Harmonic_Inversion_ExAndInternal.calculate_Bfield(data_day, Nmax_Internal=Nmax_Internal, Nmax_External=Nmax_External,
                                               path=ExAndIn_path, method=Method,Coordinate=Coordinate)
            # The LSTM Model
            B_LSTM_day = LSTM_Simulation(data_day,model_path=LSTM_path)

            B_JRM = B_In_day + B_Ex_day
            B_JRM['Btotal'] = np.sqrt(B_JRM['Br_ss'] ** 2 + B_JRM['Btheta_ss'] ** 2 + B_JRM['Bphi_ss'] ** 2)
            B_SVD = B_In_day + B_Ex_day + B_SVD_day
            B_SVD['Btotal'] = np.sqrt(B_SVD['Br_ss'] ** 2 + B_SVD['Btheta_ss'] ** 2 + B_SVD['Bphi_ss'] ** 2)
            B_LSTM = B_In_day.loc[B_LSTM_day.index] + B_Ex_day.loc[B_LSTM_day.index] + B_LSTM_day
            B_LSTM['Btotal'] = np.sqrt(B_LSTM['Br_ss'] ** 2 + B_LSTM['Btheta_ss'] ** 2 + B_LSTM['Bphi_ss'] ** 2)

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
                new_row[comp+'_SVD'] = calculate_rms_error(B_SVD[comp].values, data_day[comp].values)
                new_row[comp + '_LSTM'] = calculate_rms_error(B_LSTM[comp].values, data_day.loc[B_LSTM.index][comp].values)

            if RMS_df.empty:
                RMS_df = pd.DataFrame([new_row])
            else:
                RMS_df = pd.concat([RMS_df, pd.DataFrame([new_row])], ignore_index=True)

    plot_rms_data_Orbits(RMS_df,titles,Plot_Save_path,Nmax_Internal, Nmax_External)

def Plot_Model_Result(data_df,Model_Names=[],Save_fig = True, Fig_path = 'Result_pic/Residual_Model'):
    os.makedirs(Fig_path,exist_ok=True)
    # Setup the figure and subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))  # Adjust figsize to ensure each subplot is readable
    Orbit = Data_Date.find_orbit_by_timestamp(data_df['Br_ss'].idxmin())
    fig.suptitle(f'B residual {data_df['Br_ss'].idxmin()} Orbit {Orbit}')
    # Titles for each subplot
    titles = ['Br', 'Btheta', 'Bphi']
    y_labels = ['Br (nT)', 'Btheta (nT)', 'Bphi (nT)']

    for i, (title, y_label) in enumerate(zip(titles, y_labels)):
        # Plot actual data
        axes[i].plot(data_df.index, data_df[title + '_ss'], label='Actual ' + title, marker='o',
                     linestyle='-', markersize=4)
        # Plot predictions
        for Model in Model_Names:
            RMS = calculate_rms_error(data_df[title + '_ss' + f'_{Model}'], data_df[title + '_ss'].values)
            axes[i].plot(data_df.index, data_df[title + '_ss' + f'_{Model}'], label=f'Residual {Model}'+f' RMS={RMS:.1f}',
                         linestyle='-', markersize=4)

        # Setting titles and labels
        axes[i].set_title('Actual vs Predicted ' + title)
        axes[i].set_xlabel('Date Time')
        axes[i].set_ylabel(y_label)
        axes[i].legend()
        axes[i].grid(True)

    axes[3].plot(data_df.index, data_df['Btotal'], label='Actual Btotal', marker='o',
                 linestyle='-', markersize=4)
    # Plot predictions
    for Model in Model_Names:
        RMS = calculate_rms_error(data_df['Btotal' + f'_{Model}'], data_df['Btotal'].values)
        axes[3].plot(data_df.index, data_df['Btotal' + f'_{Model}'], label=f'Residual {Model}' +f' RMS={RMS:.1f}',
                     linestyle='-', markersize=4)

    # Setting titles and labels
    axes[3].set_title('Actual vs Predicted Btotal')
    axes[3].set_xlabel('Date Time')
    axes[3].set_ylabel('Btotal')
    axes[3].legend()
    axes[3].grid(True)
    # Adjust layout and show plot

    plt.tight_layout()
    if Save_fig:
        plt.savefig(Fig_path+f'/B residual {data_df['Br_ss'].idxmin()} Orbit {Orbit}.jpg',dpi=200)
    plt.show()

def Plot_Model_Residual_Result(data_df,Model_Names=[],Save_fig = True, Fig_path = 'Result_pic/Residual_Model'):
    os.makedirs(Fig_path,exist_ok=True)
    # Setup the figure and subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))  # Adjust figsize to ensure each subplot is readable
    Orbit = Data_Date.find_orbit_by_timestamp(data_df['Br_ss'].idxmin())
    fig.suptitle(f'B Residual of residual model {data_df['Br_ss'].idxmin()} Orbit {Orbit}')
    # Titles for each subplot
    titles = ['Br', 'Btheta', 'Bphi']
    y_labels = ['Br (nT)', 'Btheta (nT)', 'Bphi (nT)']

    for i, (title, y_label) in enumerate(zip(titles, y_labels)):
        # Plot actual data
        axes[i].plot(data_df.index, data_df[title + '_ss'], label='Actual ' + title, marker='o',
                     linestyle='-', markersize=4)
        # Plot predictions
        for Model in Model_Names:
            RMS = calculate_rms_error(data_df[title + '_ss' + f'_{Model}'], data_df[title + '_ss'].values)
            axes[i].plot(data_df.index, data_df[title + '_ss']-data_df[title + '_ss' + f'_{Model}'], label=f'Residual {Model}'+f' RMS={RMS:.1f}',
                         linestyle='-', markersize=4)

        # Setting titles and labels
        axes[i].set_title('Actual & Residual ' + title)
        axes[i].set_xlabel('Date Time')
        axes[i].set_ylabel(y_label)
        axes[i].legend()
        axes[i].grid(True)

    axes[3].plot(data_df.index, data_df['Btotal'], label='Actual Btotal', marker='o',
                 linestyle='-', markersize=4)
    # Plot predictions
    for Model in Model_Names:
        RMS = calculate_rms_error(data_df['Btotal' + f'_{Model}'], data_df['Btotal'].values)
        axes[3].plot(data_df.index, data_df['Btotal']-data_df['Btotal' + f'_{Model}'], label=f'Residual {Model}' +f' RMS={RMS:.1f}',
                     linestyle='-', markersize=4)

    # Setting titles and labels
    axes[3].set_title('Actual & Residual Btotal')
    axes[3].set_xlabel('Date Time')
    axes[3].set_ylabel('Btotal')
    axes[3].legend()
    axes[3].grid(True)
    # Adjust layout and show plot

    plt.tight_layout()
    if Save_fig:
        plt.savefig(Fig_path+f'/B Residual residual {data_df['Br_ss'].idxmin()} Orbit {Orbit}.jpg',dpi=200)
    plt.show()


def data_Add_Model(data,B_Model,Model_name):
    titles = ['Br', 'Btheta', 'Bphi']
    for title in titles:
        data[title + '_ss' + f'_{Model_name}'] = B_Model[title + '_ss']
    data['Btotal'+ f'_{Model_name}'] = B_Model['Btotal']

    return data


if __name__ == '__main__':

    ExAndIn_path = f'Spherical_Harmonic_Model/First50_Orbit_Model_ExAndInternal_24h'
    LSTM_path = 'LSTM/magnetic_field_model.h5'
    # Plot the RMS
    # Plot_RMS_Orbits(ExAndIn_path=ExAndIn_path,LSTM_path=LSTM_path)

    # PLot the B residual
    # Load data and assume you set up data as previously explained
    data = pd.read_csv('JunoFGMData/Processed_Data/LSTM_B_ResidualData_2h.csv', index_col='Time')
    data = pd.read_csv('JunoFGMData/Processed_Data/Fist_50_Orbits_Data_1s_2h.csv',index_col='Time')
    data_residual = pd.read_csv('JunoFGMData/Processed_Data/LSTM_B_ResidualData_2h.csv', index_col='Time')
    data.index = pd.to_datetime(data.index)
    # print(data.keys())
    data_residual.index = pd.to_datetime(data_residual.index)
    data_residual[['LocalTime', 'r', 'Latitude_ss', 'X_ss', 'Y_ss', 'Z_ss']] = data[['LocalTime', 'r', 'Latitude_ss', 'X_ss', 'Y_ss', 'Z_ss']]
    # print(data_residual.keys())
    Orbit = 11
    time = Data_Date.find_date_by_orbit(Orbit)

    data_residual = data_residual.loc[time]
    data = data.loc[time]

    data_residual['Btotal'] = np.sqrt(data_residual['Br_ss'] ** 2 + data_residual['Btheta_ss'] ** 2 + data_residual['Bphi_ss'] ** 2)

    '''
    
    year_doy = {'2023':[136]}
    # read data
    Data = Juno_Mag_MakeData_Function.Read_Data(year_doy, freq=1)

    # 2h data
    PeriJovian_time = Data['r'].idxmin()
    # 2 hour window data
    Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
    Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 3

    data_day = Data.loc[Time_start:Time_end]
    # data_day = data_day[::60]

    B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data_day)
    B_Ex_day = CoordinateTransform.SysIIItoSS_Bfield(data_day, B_Ex_day)
    Model = 'jrm33'
    B_In_day = Juno_Mag_MakeData_Function.MagneticField_Internal(data_day, model=Model, degree=30)
    B_In_day = CoordinateTransform.SysIIItoSS_Bfield(data_day, B_In_day)

    data_residual = data_day - B_Ex_day - B_In_day
    data_residual['Btotal'] = np.sqrt(data_residual['Br_ss'] ** 2 + data_residual['Btheta_ss'] ** 2 + data_residual['Bphi_ss'] ** 2)
    data_residual[['r_ss','theta_ss','phi_ss']] = data_day[['r_ss','theta_ss','phi_ss']]
    print(data_residual.describe())
    '''

    LSTM_Model = LSTM_Simulation(data_residual,model_path=LSTM_path)

    data_residual = data_Add_Model(data_residual,LSTM_Model,'LSTM')

    # The Residual Model
    B_SVD = Spherical_Harmonic_Inversion_ExAndInternal.calculate_Bfield(data_residual, Nmax_Internal=1,
                                                                            Nmax_External=1,
                                                                            path=ExAndIn_path, method='SVD',
                                                                            Coordinate='SS')
    data_residual = data_Add_Model(data_residual,B_SVD,'SVD')

    # Baselinemodel = BaselineModel.calculate_Bfield(data)
    # print('Baseline finished')
    # Baselinemodel = CoordinateTransform.SysIIItoSS_Bfield(data,Baselinemodel)
    #
    # B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
    # B_Ex = CoordinateTransform.SysIIItoSS_Bfield(data, B_Ex)
    # print('B_Ex finished')
    # B_baseline = data - Baselinemodel - B_Ex
    # B_baseline['Btotal'] = np.sqrt(B_baseline['Br_ss']**2+B_baseline['Btheta_ss']**2+B_baseline['Bphi_ss']**2)
    # data_residual = data_Add_Model(data_residual,B_baseline,'Baseline')

    Plot_Model_Result(data_residual,Model_Names=['LSTM','SVD'],Save_fig=True)
    Plot_Model_Residual_Result(data_residual,Model_Names=['LSTM','SVD'],Save_fig=True)