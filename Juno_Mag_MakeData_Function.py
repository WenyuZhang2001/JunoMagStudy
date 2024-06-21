#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import glob
import requests


# In[2]:


import con2020
import JupiterMag as jm


# In[3]:


import CoordinateTransform


# Constants

# In[4]:


Rj = 71492 # unit kilometers
day_1 = timedelta(days=1)
hour_1 = timedelta(hours=1)
min_1 = timedelta(minutes=1)

# Date

year_doy_pj = {'2016':[[240,1],[346,3]],
              '2017':[[33,4],[86,5],[139,6],[191,7],[244,8],[297,9],[350,10]],
              '2018':[[38,11],[91,12],[144,13],[197,14],[249,15],[302,16],[355,17]],
              '2019':[[43,18],[96,19],[149,20],[201,21],[254,22],[307,23],[360,24]],
               '2020':[[48,25],[101,26],[154,27],[207,28],[259,29],[312,30],[365,31]],
               '2021':[[52,32],[105,33],[159,34],[202,35],[245,36],[289,37],[333,38]],
               '2022':[[12,39],[55,40],[99,41],[142,42],[186,43],[229,44],[272,45],[310,46],[348,47]],
               '2023':[[22,48],[60,49],[98,50]]}

# Functions

def dateList(year_doy):
    date_list = pd.DataFrame(columns=['year','doy'])

    for year in year_doy.keys():
        for doy in year_doy[year]:
            date_list = pd.concat([date_list,pd.DataFrame({'year':[year],'doy':[doy]})],
                                  ignore_index=True)


    date_list['Time'] = date_list['year']+'-'+date_list['doy'].map(str)
    date_list['Time'] = pd.to_datetime(date_list['Time'],format='%Y-%j')

    return date_list


def change_stsTocsv(directory_path=''):

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # List all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file ends with .sts
        if filename.endswith('.sts'):
            # Construct the full file path
            old_file = os.path.join(directory_path, filename)
            # Replace .sts with .vsc in the file name
            new_file = os.path.join(directory_path, filename[:-4] + '.csv')
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')

def JupiterMagExternal(Data_1,Data_2,Data_3,cartesianIn = False,cartesianOut = False):
    # initial model
    External_model = con2020.Model(CartesianIn=cartesianIn,CartesianOut=cartesianOut)

    # calculate
    B = External_model.Field(Data_1,Data_2,Data_3)
    return B

def JupiterMagInternal(Data_1,Data_2,Data_3,model = 'jrm33',cartesianIn = False,cartesianOut = False,degree=10):
    # initial model
    jm.Internal.Config(Model=model,CartesianIn=cartesianIn,CartesianOut=cartesianOut,Degree=degree)

    #calculate
    B_1,B_2,B_3 = jm.Internal.Field(Data_1,Data_2,Data_3)

    return B_1,B_2,B_3

def data_changeIndex(data):
    data['Time'] = data['Year'].map(str) + '-' + data['Doy'].map(str) + '-' + data['Hour'].map(str) \
                   + '-' + data['Min'].map(str) + '-' + data['Sec'].map(str) + '-' + data['Msec'].map(str)
    data['Time'] = pd.to_datetime(data['Time'], format='%Y-%j-%H-%M-%S-%f')
    data = data.set_index('Time')
    return data
def data_process(data,Rj = 71492):
    data['X'] = data['X']/Rj
    data['Y'] = data['Y']/Rj
    data['Z'] = data['Z']/Rj
    data['Xss'] = data['Xss'] / Rj
    data['Yss'] = data['Yss'] / Rj
    data['Zss'] = data['Zss'] / Rj

    r,theta,phi = CoordinateTransform.CartesiantoSpherical(data['X'].to_numpy(),
                                                           data['Y'].to_numpy(),
                                                           data['Z'].to_numpy())
    Lat = 90-theta
    Long = np.where(phi<0,360+phi,phi)

    data['r'] = r
    data['theta'] = theta
    data['phi'] = phi
    data['Latitude'] = Lat
    data['Longitude'] = Long

    r_ss, theta_ss, phi_ss = CoordinateTransform.CartesiantoSpherical(data['Xss'].to_numpy(),
                                                             data['Yss'].to_numpy(),
                                                             data['Zss'].to_numpy())
    Lat_ss = 90 - theta_ss
    Long_ss = np.where(phi_ss < 0, 360 + phi_ss, phi_ss)
    LocalTime = Long_ss*24/360 + 12.0
    LocalTime = np.where(LocalTime >= 24,LocalTime-24,LocalTime)
    data['r_ss'] = r_ss
    data['theta_ss'] = theta_ss
    data['phi_ss'] = phi_ss
    data['Latitude_ss'] = Lat_ss
    data['Longitude_ss'] = Long_ss
    data['LocalTime'] = LocalTime

    # data['Time'] = data['Year'].map(str)+'-'+data['Doy'].map(str)+'-'+data['Hour'].map(str)\
    #              +'-'+data['Min'].map(str)+'-'+data['Sec'].map(str)+'-'+data['Msec'].map(str)
    # data['Time'] = pd.to_datetime(data['Time'],format='%Y-%j-%H-%M-%S-%f')
    # data = data.set_index('Time')

    data['Bx'] = data['Bx'].map(float)
    data['By'] = data['By'].map(float)
    data['Bz'] = data['Bz'].map(float)
    data['Btotal'] = (data['Bx']**2+data['By']**2+data['Bz']**2)**0.5
    data['BxSS'] = data['BxSS'].map(float)
    data['BySS'] = data['BySS'].map(float)
    data['BzSS'] = data['BzSS'].map(float)
    data['BtotalSS'] = (data['BxSS'] ** 2 + data['BySS'] ** 2 + data['BzSS'] ** 2) ** 0.5

    Br,Btheta,Bphi = CoordinateTransform.CartesiantoSpherical_Bfield(data['X'].to_numpy(),
                                                                    data['Y'].to_numpy(),
                                                                    data['Z'].to_numpy(),
                                                                    data['Bx'].to_numpy(),
                                                                    data['By'].to_numpy(),
                                                                    data['Bz'].to_numpy())
    data['Br'] = Br
    data['Bphi'] = Bphi
    data['Btheta'] = Btheta

    Br_ss, Btheta_ss, Bphi_ss = CoordinateTransform.CartesiantoSpherical_Bfield(data['Xss'].to_numpy(),
                                                                       data['Yss'].to_numpy(),
                                                                       data['Zss'].to_numpy(),
                                                                       data['BxSS'].to_numpy(),
                                                                       data['BySS'].to_numpy(),
                                                                       data['BzSS'].to_numpy())
    data['Br_ss'] = Br_ss
    data['Bphi_ss'] = Bphi_ss
    data['Btheta_ss'] = Btheta_ss

    return data



def Read_Data(year_doy,directory_path = 'JunoFGMData/',freq=1):
    change_stsTocsv(directory_path)
    col_names_pc = ['Year','Doy','Hour','Min','Sec','Msec','DDay','Bx','By','Bz','Range','X','Y','Z']
    col_names_ss = ['Year','Doy','Hour','Min','Sec','Msec','DDay','BxSS','BySS','BzSS','Range','Xss','Yss','Zss']
    col_names = ['Bx','By','Bz','X','Y','Z','BxSS','BySS','BzSS','Xss','Yss','Zss']
    data = pd.DataFrame(columns=col_names)
    for year in year_doy.keys():
        for doy in year_doy[year]:

            pattern_pc = directory_path + f'fgm_jno_l3_{year}{doy:0>3d}pc_*r{freq}s_*.csv'
            pattern_ss = directory_path + f'fgm_jno_l3_{year}{doy:0>3d}ss_*r{freq}s_*.csv'

            file_list_pc = glob.glob(pattern_pc)
            file_list_ss = glob.glob(pattern_ss)

            dataframes_list_pc = []
            dataframes_list_ss = []
            # Iterate through the list of file names

            # pc coordinate
            for file_name in file_list_pc:
                print('read data file: '+file_name)
                # Read each CSV file into a DataFrame
                temp_data = pd.read_csv(file_name,header = None,sep = '\s+',names=col_names_pc,index_col=False)
                temp_data = temp_data.dropna()
                temp_data = data_changeIndex(temp_data)
                temp_data.drop(['Year', 'Doy', 'Hour', 'Min', 'Sec', 'Msec', 'DDay', 'Range'], axis=1, inplace=True)

                # Append the DataFrame to the list
                dataframes_list_pc.append(temp_data)

            # ss coordinate
            for file_name in file_list_ss:
                print('read data file: ' + file_name)
                # Read each CSV file into a DataFrame
                temp_data = pd.read_csv(file_name, header=None, sep='\s+', names=col_names_ss, index_col=False)
                temp_data = temp_data.dropna()
                temp_data = data_changeIndex(temp_data)
                temp_data.drop(['Year','Doy','Hour','Min','Sec','Msec','DDay','Range'],axis=1,inplace=True)

                # Append the DataFrame to the list
                dataframes_list_ss.append(temp_data)


            # Optionally, combine all DataFrames into a single DataFrame
            try:
                combined_df_pc = pd.concat(dataframes_list_pc)
                combined_df_ss = pd.concat(dataframes_list_ss)
                combined_df = pd.concat([combined_df_pc,combined_df_ss],axis=1,join='inner')
            except:
                print(f'No File in the doy {year}{doy}')
                continue

            if data.empty:
                data = combined_df
            else:
                data = pd.concat([data,combined_df])

    # process data 
    # simple caculations and unit transform
    data = data_process(data)
    data = data.sort_index()

    if not data.empty:
        print(f'Data Time: {data.index[0]} - {data.index[-1]}')
    else:
        print('No data on those days!')
    return data




def MagneticField_Internal(data,model='jrm33',degree=10):

    # Internal
    BxIn,ByIn,BzIn = JupiterMagInternal(data['X'].to_numpy(),
                                      data['Y'].to_numpy(),
                                      data['Z'].to_numpy(),
                                        model = model,
                                      cartesianIn=True,cartesianOut=True,degree=degree)
    B_In_np = np.vstack((BxIn,ByIn,BzIn)).T
    B_In = pd.DataFrame(B_In_np,columns=['Bx','By','Bz'],index=data['X'].index)
    B_In['Btotal'] = (B_In['Bx']**2 + B_In['By']**2 + B_In['Bz']**2)**0.5

    Br,Btheta,Bphi = CoordinateTransform.CartesiantoSpherical_Bfield(data['X'].to_numpy(),
                                                                    data['Y'].to_numpy(),
                                                                    data['Z'].to_numpy(),
                                                                    B_In['Bx'].to_numpy(),
                                                                    B_In['By'].to_numpy(),
                                                                    B_In['Bz'].to_numpy())

    B_In['Br'] = Br
    B_In['Bphi'] = Bphi
    B_In['Btheta'] = Btheta

    return B_In

def MagneticField_External(data):

    # External
    B_Ex_np = JupiterMagExternal(data['X'].to_numpy(),
                                      data['Y'].to_numpy(),
                                      data['Z'].to_numpy(),
                                      cartesianIn=True,cartesianOut=True)

    B_Ex = pd.DataFrame(B_Ex_np,columns=['Bx','By','Bz'],index=data['X'].index)
    B_Ex['Btotal'] = (B_Ex['Bx']**2 + B_Ex['By']**2 + B_Ex['Bz']**2)**0.5

    Br,Btheta,Bphi = CoordinateTransform.CartesiantoSpherical_Bfield(data['X'].to_numpy(),
                                                                    data['Y'].to_numpy(),
                                                                    data['Z'].to_numpy(),
                                                                    B_Ex['Bx'].to_numpy(),
                                                                    B_Ex['By'].to_numpy(),
                                                                    B_Ex['Bz'].to_numpy())

    B_Ex['Br'] = Br
    B_Ex['Bphi'] = Bphi
    B_Ex['Btheta'] = Btheta

    return B_Ex

def JupiterMagFieldLineTrace(x,y,z,extmodel='none',traceDir=0,maxLen=5000):

    jm.Con2020.Config(equation_type='analytic')

    FLT = jm.TraceField(x,y,z,Verbose=False,IntModel='jrm33',ExtModel=extmodel,TraceDir=traceDir,MaxLen=maxLen)


    return FLT

def FootPrintCalculate(data,Extmodel='con2020',maxLen=5000):

    data_FLT = JupiterMagFieldLineTrace(data['X'],data['Y'],data['Z'],extmodel=Extmodel,maxLen=maxLen)

    North_FP_X,North_FP_Y,North_FP_Z = data_FLT.x[:,0],data_FLT.y[:,0],data_FLT.z[:,0]
    South_FP_X,South_FP_Y,South_FP_Z = data_FLT.x[range(len(data_FLT.x)),np.isnan(data_FLT.x).argmax(axis=1)-1],\
                                       data_FLT.y[range(len(data_FLT.x)),np.isnan(data_FLT.y).argmax(axis=1)-1],\
                                       data_FLT.z[range(len(data_FLT.x)),np.isnan(data_FLT.z).argmax(axis=1)-1]
    North_FP_r,North_FP_theta,North_FP_phi = CoordinateTransform.CartesiantoSpherical(North_FP_X,North_FP_Y,
                                                                                      North_FP_Z)
    South_FP_r,South_FP_theta,South_FP_phi = CoordinateTransform.CartesiantoSpherical(South_FP_X,South_FP_Y,
                                                                                      South_FP_Z)
    North_FP_Lat = 90.0 - North_FP_theta
    South_FP_Lat = 90.0 - South_FP_theta
    North_FP_Long = np.where(North_FP_phi<0,360+North_FP_phi,North_FP_phi)
    South_FP_Long = np.where(South_FP_phi<0,360+South_FP_phi,South_FP_phi)
    North_FP_ArcLen = np.pi/2 - North_FP_Lat*2*np.pi/360
    South_FP_ArcLen = np.pi/2 + South_FP_Lat*2*np.pi/360

    Juno_MAG_FP = pd.DataFrame()
    Juno_MAG_FP['North_FP_Lat'] = North_FP_Lat
    Juno_MAG_FP['South_FP_Lat'] = South_FP_Lat
    Juno_MAG_FP['North_FP_Long'] = North_FP_Long
    Juno_MAG_FP['South_FP_Long'] = South_FP_Long
    Juno_MAG_FP['South_FP_ArcLen'] = South_FP_ArcLen
    Juno_MAG_FP['North_FP_ArcLen'] = North_FP_ArcLen
    Juno_MAG_FP.index = data.index
    Juno_MAG_FP['r'] = data['r']

    return Juno_MAG_FP

def MinLat(Juno_MAG_FP,pj=99):
    if pj == 99:
        print('Wrong PJ input!')
        return
    Juno_MAG_FP['PJ'] = np.ones(len(Juno_MAG_FP))*pj

    try:
        Juno_MAG_FP_North = pd.read_csv('Result_data/Juno_MAG_FP_MinLatitudeNorth.csv',index_col=0)
        Juno_MAG_FP_North = pd.concat([Juno_MAG_FP_North,
                                       Juno_MAG_FP[Juno_MAG_FP['North_FP_Lat']==Juno_MAG_FP['North_FP_Lat'].min()]])
        Juno_MAG_FP_North.to_csv('Result_data/Juno_MAG_FP_MinLatitudeNorth.csv')
    except:
        Juno_MAG_FP_North = Juno_MAG_FP[Juno_MAG_FP['North_FP_Lat']==Juno_MAG_FP['North_FP_Lat'].min()]
        Juno_MAG_FP_North.to_csv('Result_data/Juno_MAG_FP_MinLatitudeNorth.csv')

    try:
        Juno_MAG_FP_South = pd.read_csv('Result_data/Juno_MAG_FP_MinLatitudeSouth.csv',index_col=0)
        Juno_MAG_FP_South = pd.concat([Juno_MAG_FP_South,
                                       Juno_MAG_FP[Juno_MAG_FP['South_FP_Lat']==Juno_MAG_FP['South_FP_Lat'].max()]])
        Juno_MAG_FP_South.to_csv('Result_data/Juno_MAG_FP_MinLatitudeSouth.csv')
    except:
        Juno_MAG_FP_South = Juno_MAG_FP[Juno_MAG_FP['South_FP_Lat']==Juno_MAG_FP['South_FP_Lat'].max()]
        Juno_MAG_FP_South.to_csv('Result_data/Juno_MAG_FP_MinLatitudeSouth.csv')

def Max_Delta_Bfield_Btotal(data,B_In,B_Ex, pj=99,Coordinate='Sys3'):
    if pj == 99:
        print('Wrong PJ input!')
        return
    data['PJ'] = np.ones(len(data)) * pj
    if Coordinate=='Sys3':
        # Calculate
        data['delta_Bx'] = np.array(np.abs(data['Bx']-B_In['Bx']-B_Ex['Bx']))
        data['delta_By'] = np.abs(data['By']-B_In['By']-B_Ex['By'])
        data['delta_Bz'] = np.abs(data['Bz']-B_In['Bz']-B_Ex['Bz']).T
        data['delta_Btotal'] = np.abs(data['Btotal']-B_In['Btotal']-B_Ex['Btotal'])
        data['Btotal_delta'] = np.sqrt(data['delta_Bx']**2+data['delta_By']**2+data['delta_Bz']**2)


        try:
            Juno_MAG_Max_delta_Btotal = pd.read_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv',index_col=0)
            Juno_MAG_Max_delta_Btotal = pd.concat([Juno_MAG_Max_delta_Btotal,data.loc[data['delta_Btotal'].idxmax()].to_frame().T])
            Juno_MAG_Max_delta_Btotal.to_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv')
        except:
            Juno_MAG_Max_delta_Btotal = data.loc[data['delta_Btotal'].idxmax()].to_frame().T
            print(Juno_MAG_Max_delta_Btotal)
            Juno_MAG_Max_delta_Btotal.to_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv')
        try:
            Juno_MAG_Max_Btotal_delta = pd.read_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv',index_col=0)
            Juno_MAG_Max_Btotal_delta = pd.concat([Juno_MAG_Max_Btotal_delta,data.loc[data['Btotal_delta'].idxmax()].to_frame().T])
            Juno_MAG_Max_Btotal_delta.to_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv')
        except:
            Juno_MAG_Max_Btotal_delta = data.loc[data['Btotal_delta'].idxmax()].to_frame().T
            Juno_MAG_Max_Btotal_delta.to_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv')

    elif Coordinate =='Spherical':
        data['delta_Br'] = np.abs(data['Br'] - B_In['Br'] - B_Ex['Br'])
        data['delta_Btheta'] = np.abs(data['Btheta'] - B_In['Btheta'] - B_Ex['Btheta'])
        data['delta_Bphi'] = np.abs(data['Bphi'] - B_In['Bphi'] - B_Ex['Bphi'])
        data['delta_Btotal'] = np.abs(data['Btotal'] - B_In['Btotal'] - B_Ex['Btotal'])
        data['Btotal_delta'] = np.sqrt(data['delta_Br'] ** 2 + data['delta_Btheta'] ** 2 + data['delta_Bphi'] ** 2)

        try:
            Juno_MAG_Max_delta_Btotal = pd.read_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv',index_col=0)
            Juno_MAG_Max_delta_Btotal = pd.concat([Juno_MAG_Max_delta_Btotal,data.loc[data['delta_Btotal'].idxmax()].to_frame().T])
            Juno_MAG_Max_delta_Btotal.to_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv')
        except:
            Juno_MAG_Max_delta_Btotal = data.loc[data['delta_Btotal'].idxmax()].to_frame().T
            Juno_MAG_Max_delta_Btotal.to_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv')
        try:
            Juno_MAG_Max_Btotal_delta = pd.read_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv',index_col=0)
            Juno_MAG_Max_Btotal_delta = pd.concat([Juno_MAG_Max_Btotal_delta,data.loc[data['Btotal_delta'].idxmax()].to_frame().T])
            Juno_MAG_Max_Btotal_delta.to_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv')
        except:
            Juno_MAG_Max_Btotal_delta = data.loc[data['delta_Btotal'].idxmax()].to_frame().T
            Juno_MAG_Max_Btotal_delta.to_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv')

def read_24hData(year_doy_pj,freq=60):
    data = pd.DataFrame()

    for year in year_doy_pj.keys():
        for doy in year_doy_pj[year]:

            year_doy = {year:[doy[0]]}
            date_list = dateList(year_doy)

            # read data
            Data = Read_Data(year_doy,freq=freq)

            # 24 hours data
            Time_start = date_list['Time'].iloc[0]
            Time_end = Time_start+hour_1*24

            data_day = Data.loc[Time_start:Time_end]

            if data.empty:
                data = data_day
            else:
                data = pd.concat([data, data_day])
    return data

def Caluclate_B_Residual(data,B_In,B_Ex):
    '''

    :param data: MAG data
    :param B_In: B internal
    :param B_Ex: B External
    :return: B Residual Field
    '''

    B_residual = pd.DataFrame(index = data.index)
    component = 'Br'
    B_residual[component] = data[component]-B_In[component]-B_Ex[component]
    component = 'Btheta'
    B_residual[component] = data[component] - B_In[component] - B_Ex[component]
    component = 'Bphi'
    B_residual[component] = data[component] - B_In[component] - B_Ex[component]
    component = 'Bx'
    B_residual[component] = data[component] - B_In[component] - B_Ex[component]
    component = 'By'
    B_residual[component] = data[component] - B_In[component] - B_Ex[component]
    component = 'Bz'
    B_residual[component] = data[component] - B_In[component] - B_Ex[component]
    component = 'Btotal'
    B_residual[component] = data[component] - B_In[component] - B_Ex[component]

    return  B_residual