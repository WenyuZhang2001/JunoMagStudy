#!/usr/bin/env python
# coding: utf-8

# In[1]:


import Juno_Mag_MakeData_Function
import pandas as pd
import numpy as np

import Spherical_Harmonic_InversionModel_Functions

# In[2]:


year_doy_pj = {'2016':[[240,1],[346,3]],
              '2017':[[33,4],[86,5],[139,6],[191,7],[244,8],[297,9],[350,10]],
              '2018':[[38,11],[91,12],[144,13],[197,14],[249,15],[302,16],[355,17]],
              '2019':[[43,18],[96,19],[149,20],[201,21],[254,22],[307,23],[360,24]],
               '2020':[[48,25],[101,26],[154,27],[207,28],[259,29],[312,30],[365,31]],
               '2021':[[52,32],[105,33],[159,34],[202,35],[245,36],[289,37],[333,38]],
               '2022':[[12,39],[55,40],[99,41],[142,42],[186,43],[229,44],[272,45],[310,46],[348,47]],
               '2023':[[22,48],[60,49],[98,50]]}


# In[3]:


def Save_60sData(year_doy_pj):
    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            # read data     
            data = Juno_Mag_MakeData_Function.Read_Data(year_doy,freq=60)
            # FLT
            Juno_MAG_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(data)

            data = pd.concat([data,Juno_MAG_FP],axis=1)
            data['PJ'] = np.ones(len(data))*pj
            # save
            data.to_csv(f'JunoFGMData/Processed_Data/JunoMAG_FLT_60s_{year}{doy[0]:0>3d}_PJ{pj:0>2d}.csv')
            print(f'Data on {year}{doy[0]:0>3d}_PJ{pj:0>2d} saved')


# In[4]:


def Save_1sData(year_doy_pj):
    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            pj = doy[1]
            year_doy = {year:[doy[0]]}
            # read data     
            data = Juno_Mag_MakeData_Function.Read_Data(year_doy,freq=1)
            # FLT
            Juno_MAG_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(data)

            data = pd.concat([data,Juno_MAG_FP],axis=1)
            data['PJ'] = np.ones(len(data))*pj
            # save
            data.to_csv(f'JunoFGMData/Processed_Data/JunoMAG_FLT_1s_{year}{doy[0]:0>3d}_PJ{pj:0>2d}.csv')
            print(f'Data on {year}{doy[0]:0>3d}_PJ{pj:0>2d} saved')


# In[5]:


# Save_60sData(year_doy_pj)


# In[6]:


# Save_1sData(year_doy_pj)

# Save first 20 Orbits and B_Ex and B_In

# Model Compared to
Model = 'jrm33'

# read the data
def read_data(year_doy_pj):
    data = pd.DataFrame()
    B_In = pd.DataFrame()
    B_Ex = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:
            # pj = doy[1]
            year_doy = {year:[doy[0]]}
            # date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

            # read data
            Data = Juno_Mag_MakeData_Function.Read_Data(year_doy,freq=1)

            # 24 hours data
            Time_start = Data.index.min()
            Time_end = Data.index.max()
            # Time_end = Time_start+Juno_Mag_MakeData_Function.hour_1*24

            # Check the periJovian point time
            PeriJovian_time = Data['r'].idxmin()
            # 2 hour window data
            Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
            Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 2

            data_day = Data.loc[Time_start:Time_end]
            # data_day = data_day[::60]

            B_Ex_day = Juno_Mag_MakeData_Function.MagneticField_External(data_day)
            B_In_day = Juno_Mag_MakeData_Function.MagneticField_Internal(data_day, model=Model, degree=30)

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])

            if B_Ex.empty:
                B_Ex = B_Ex_day
            else:
                B_Ex = pd.concat([B_Ex,B_Ex_day])

            if B_In.empty:
                B_In = B_In_day
            else:
                B_In = pd.concat([B_In,B_In_day])

    return data ,B_In,B_Ex

def Save_B_Residual_LSTM(data,B_Residual):

    data_set = pd.DataFrame()
    data['Time'] = pd.to_datetime(data['Time'])
    data.index = data['Time']
    B_Residual.index = data.index
    data_set.index = data.index
    data_set[['r','theta','phi','r_ss','theta_ss','phi_ss','LocalTime']] = data[['r','theta','phi','r_ss','theta_ss','phi_ss','LocalTime']]
    data_set[['Br_ss','Btheta_ss','Bphi_ss']] = B_Residual[['Br_ss','Btheta_ss','Bphi_ss']]
    data_set[['Br','Btheta','Bphi','Btotal']] = B_Residual[['Br','Btheta','Bphi','Btotal']]

    data_set.to_csv('JunoFGMData/Processed_Data/LSTM_B_ResidualData_2h.csv')
    print(data_set)
    return

if __name__ == '__main__':

    # data, B_In, B_Ex = read_data(year_doy_pj)
    # data = read_data(year_doy_pj)
    # data.to_csv('JunoFGMData/Processed_Data/First_50_Orbits_Data_1s_2h.csv')
    # data.to_csv('JunoFGMData/Processed_Data/First_50_Orbits_Data_60s_24h.csv')
    # print('Data read and saved')


    # B_Ex.to_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Ex_1s_2h.csv')
    # B_Ex.to_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Ex_60s_24h.csv')
    # print('External Field calculated and saved')
    #
    # B_In.to_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_In_1s_2h.csv')
    # B_In.to_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_In_60s_24h.csv')
    # print('Internal field calculated and saved')

    # data = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_Data_60s_24h.csv')
    data = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_Data_1s_2h.csv')
    # # B_Residual = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Residual_60s_24h.csv')
    B_Residual = pd.read_csv('JunoFGMData/Processed_Data/First_50_Orbits_B_Residual_1s_2h.csv')
    # print(B_Residual.keys())

    Save_B_Residual_LSTM(data,B_Residual)


