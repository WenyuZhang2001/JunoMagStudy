#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import CoordinateTransform
import con2020
import JupiterMag as jm
import Juno_Mag_MakeData_Function
import MyPlot_Functions
import Spherical_Harmonic_InversionModel_Functions

# # Each PeriJovian Doy


#
# year_doy_pj = {'2016':[[240,1],[346,3]],
#               '2017':[[33,4],[86,5],[139,6],[191,7],[244,8],[297,9],[350,10]],
#               '2018':[[38,11],[91,12],[144,13],[197,14],[249,15],[302,16],[355,17]],
#               '2019':[[43,18],[96,19],[149,20],[201,21],[254,22],[307,23],[360,24]],
#                '2020':[[48,25],[101,26],[154,27],[207,28],[259,29],[312,30],[365,31]],
#                '2021':[[52,32],[105,33],[159,34],[202,35],[245,36],[289,37],[333,38]],
#                '2022':[[12,39],[55,40],[99,41],[142,42],[186,43],[229,44],[272,45],[310,46],[348,47]],
#                '2023':[[22,48],[60,49],[98,50]]}


year_doy_pj = {'2017':[[297,9],[350,10]]}

# # Parameters



# # Make Plots


'''
for year in year_doy_pj.keys():
    for doy in year_doy_pj[year]:
        pj = doy[1]
        year_doy = {year:[doy[0]]}
        date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

        # Make Dir
        filename = f'Result_pic/Juno_Orbit_{pj:0>2d}'
        os.makedirs(filename,exist_ok=True)
        filename = f'Result_pic/Juno_Orbit_{pj:0>2d}/Sys3'
        os.makedirs(filename,exist_ok=True)
        filename = f'Result_pic/Juno_Orbit_{pj:0>2d}/Spherical'
        os.makedirs(filename,exist_ok=True)
        filename = f'Result_pic/Juno_Orbit_{pj:0>2d}/Juno_FP'
        os.makedirs(filename,exist_ok=True)

        # read data
        Data = Juno_Mag_MakeData_Function.Read_Data_60s(year_doy)

        # 24 hours data
        Time_start = date_list['Time'].iloc[0]
        Time_end = Time_start+Juno_Mag_MakeData_Function.hour_1*24

        data = Data.loc[Time_start:Time_end]

        B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
        B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data,model=Model)

        Juno_MAG_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(data)

        # Plot
        MyPlot_Functions.Plot_Juno_Position(data=data,ShowPlot=False,pj=pj)

        Coord = 'Sys3'
        MyPlot_Functions.Plot_Bfeild(data=data,B_In=B_In,B_Ex=B_Ex,Coordinate=Coord,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=False,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=True,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      Savefig=True,ShowPlot=False,Percentage_ylim=0,Model=Model)

        Coord = 'Spherical'
        MyPlot_Functions.Plot_Bfeild(data=data,B_In=B_In,B_Ex=B_Ex,Coordinate=Coord,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=False,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=True,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      Savefig=True,ShowPlot=False,Percentage_ylim=0,Model=Model)

        MyPlot_Functions.Plot_Juno_Footprint(Juno_MAG_FP,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Juno_Footprint_thorughTime(Juno_MAG_FP,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Juno_Footprint_Anomaly(Juno_MAG_FP, pj=pj, ShowPlot=False)
        
        # Find Min Latitude and save to file
        Juno_Mag_MakeData_Function.MinLat(Juno_MAG_FP,pj=pj)

        # Check the periJovian point time
        PeriJovian_time = Data['r'].idxmin()
        # 2 hour window data
        Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1*1
        Time_end = Time_start+Juno_Mag_MakeData_Function.hour_1*3

        data = Data.loc[Time_start:Time_end]

        B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
        B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data,model=Model)

        Juno_MAG_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(data)
        
        # find delta Btotal max and Btotal delta max
        Juno_Mag_MakeData_Function.Max_Delta_Bfield_Btotal(data,B_In,B_Ex,pj=pj,Coordinate='Sys3')
        Juno_Mag_MakeData_Function.Max_Delta_Bfield_Btotal(data,B_In,B_Ex,pj=pj,Coordinate='Spherical')
        
        # Plot
        MyPlot_Functions.Plot_Juno_Position(data=data,ShowPlot=False,pj=pj)

        Coord = 'Sys3'
        MyPlot_Functions.Plot_Bfeild(data=data,B_In=B_In,B_Ex=B_Ex,Coordinate=Coord,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=False,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=True,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      Savefig=True,ShowPlot=False,Percentage_ylim=0,Model=Model)

        Coord = 'Spherical'
        MyPlot_Functions.Plot_Bfeild(data=data,B_In=B_In,B_Ex=B_Ex,Coordinate=Coord,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=False,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      MinusBex=True,Savefig=True,ShowPlot=False,Percentage_ylim=10,Model=Model)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data,B_In=B_In,B_Ex=B_Ex,pj=pj,Coordinate=Coord,
                      Savefig=True,ShowPlot=False,Percentage_ylim=0,Model=Model)

        MyPlot_Functions.Plot_Juno_Footprint(Juno_MAG_FP,pj=pj,ShowPlot=False)
        MyPlot_Functions.Plot_Juno_Footprint_thorughTime(Juno_MAG_FP, pj=pj, ShowPlot=False)
        MyPlot_Functions.Plot_Juno_Footprint_Anomaly(Juno_MAG_FP, pj=pj, ShowPlot=False)
        print(f'Plot of Pj{pj} {year_doy} is finished'+'\n'+'='*50)
'''

for year in year_doy_pj.keys():
    for doy in year_doy_pj[year]:
        pj = doy[1]
        year_doy = {year: [doy[0]]}
        date_list = Juno_Mag_MakeData_Function.dateList(year_doy)

        Model = 'SVD'
        # Model = 'jrm33'

        # path = 'Spherical_Harmonic_Model/'
        # path=''
        path = 'Spherical_Harmonic_Model/First50_Orbit_Model'
        # Make Dir
        filename = f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}'
        os.makedirs(filename, exist_ok=True)
        filename = f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/Sys3'
        os.makedirs(filename, exist_ok=True)
        filename = f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/Spherical'
        os.makedirs(filename, exist_ok=True)
        filename = f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/Juno_FP'
        os.makedirs(filename, exist_ok=True)

        # read data
        Data = Juno_Mag_MakeData_Function.Read_Data(year_doy,freq=1)
        Data = Data.iloc[::60]
        # # Doing the Rc=0.85
        # Data['r'] = Data['r']/0.85

        # 24 hours data
        Time_start = date_list['Time'].iloc[0]
        Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 24

        data = Data.loc[Time_start:Time_end]

        B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
        # B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model,degree=20)
        B_In = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=40, path=path,method=Model)

        # Juno_MAG_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(data)

        # Plot
        MyPlot_Functions.Plot_Juno_Position(data=data, ShowPlot=False, pj=pj,path=path)

        Coord = 'Sys3'
        MyPlot_Functions.Plot_Bfeild(data=data, B_In=B_In, B_Ex=B_Ex, Coordinate=Coord, pj=pj, ShowPlot=False,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=False, Savefig=True, ShowPlot=False, Percentage_ylim=10,
                                           Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=True, Savefig=True, ShowPlot=False, Percentage_ylim=10, Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                                  Savefig=True, ShowPlot=False, Percentage_ylim=0, Model=Model,path=path)

        Coord = 'Spherical'
        MyPlot_Functions.Plot_Bfeild(data=data, B_In=B_In, B_Ex=B_Ex, Coordinate=Coord, pj=pj, ShowPlot=False,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=False, Savefig=True, ShowPlot=False, Percentage_ylim=10,
                                           Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=True, Savefig=True, ShowPlot=False, Percentage_ylim=10, Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                                  Savefig=True, ShowPlot=False, Percentage_ylim=0, Model=Model,path=path)

        # MyPlot_Functions.Plot_Juno_Footprint(Juno_MAG_FP, pj=pj, ShowPlot=False,path=path)
        # MyPlot_Functions.Plot_Juno_Footprint_thorughTime(Juno_MAG_FP, pj=pj, ShowPlot=False,path=path)
        # MyPlot_Functions.Plot_Juno_Footprint_Anomaly(Juno_MAG_FP, pj=pj, ShowPlot=False,path=path)

        # Find Min Latitude and save to file
        # Juno_Mag_MakeData_Function.MinLat(Juno_MAG_FP, pj=pj)

        # Check the periJovian point time
        PeriJovian_time = Data['r'].idxmin()
        # 2 hour window data
        Time_start = PeriJovian_time - Juno_Mag_MakeData_Function.hour_1 * 1
        Time_end = Time_start + Juno_Mag_MakeData_Function.hour_1 * 3

        data = Data.loc[Time_start:Time_end]

        B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)
        # B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data, model=Model,degree=20)
        B_In = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=40, path=path,method=Model)

        # Juno_MAG_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(data)

        # find delta Btotal max and Btotal delta max
        Juno_Mag_MakeData_Function.Max_Delta_Bfield_Btotal(data, B_In, B_Ex, pj=pj, Coordinate='Sys3')
        Juno_Mag_MakeData_Function.Max_Delta_Bfield_Btotal(data, B_In, B_Ex, pj=pj, Coordinate='Spherical')

        # Plot
        MyPlot_Functions.Plot_Juno_Position(data=data, ShowPlot=False, pj=pj,path=path)

        Coord = 'Sys3'
        MyPlot_Functions.Plot_Bfeild(data=data, B_In=B_In, B_Ex=B_Ex, Coordinate=Coord, pj=pj, ShowPlot=False,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=False, Savefig=True, ShowPlot=False, Percentage_ylim=10,
                                           Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=True, Savefig=True, ShowPlot=False, Percentage_ylim=10, Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                                  Savefig=True, ShowPlot=False, Percentage_ylim=0, Model=Model,path=path)

        Coord = 'Spherical'
        MyPlot_Functions.Plot_Bfeild(data=data, B_In=B_In, B_Ex=B_Ex, Coordinate=Coord, pj=pj, ShowPlot=False,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=False, Savefig=True, ShowPlot=False, Percentage_ylim=10,
                                           Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                           MinusBex=True, Savefig=True, ShowPlot=False, Percentage_ylim=10, Model=Model,path=path)
        MyPlot_Functions.Plot_Delta_Bfield_Btotal(data=data, B_In=B_In, B_Ex=B_Ex, pj=pj, Coordinate=Coord,
                                                  Savefig=True, ShowPlot=False, Percentage_ylim=0, Model=Model,path=path)

        # MyPlot_Functions.Plot_Juno_Footprint(Juno_MAG_FP, pj=pj, ShowPlot=False,path=path)
        # MyPlot_Functions.Plot_Juno_Footprint_thorughTime(Juno_MAG_FP, pj=pj, ShowPlot=False,path=path)
        # MyPlot_Functions.Plot_Juno_Footprint_Anomaly(Juno_MAG_FP, pj=pj, ShowPlot=False,path=path)
        print(f'Plot of Pj{pj} {year_doy} is finished' + '\n' + '=' * 50)



# End




