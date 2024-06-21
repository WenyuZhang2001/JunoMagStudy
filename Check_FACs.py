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
import matplotlib.dates as mdates

def main():

    # import date
    # year_doy_pj = Juno_Mag_MakeData_Function.year_doy_pj
    # For test
    # year_doy_pj = {'2017':[[139,6]]}
    year_doy_pj = {'2020':[[48,25]]}
    # data
    # read 1 s data
    data = Juno_Mag_MakeData_Function.read_24hData(year_doy_pj,freq=1)

    # doing resmapling, 1s => 60s
    data = data.iloc[::10]

    # Calculate the B field Internal and External
    # Internal Use my Spherical Harmonic Model

    Model_Path = 'Spherical_Harmonic_Model/First50_Orbit_Model'
    Model = 'SVD'
    Nmax = 40
    B_In = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data, Nmax=Nmax, path=Model_Path, method=Model)

    # External Use Con2020
    B_Ex = Juno_Mag_MakeData_Function.MagneticField_External(data)

    # calculate the B field residual

    B_residual = Juno_Mag_MakeData_Function.Caluclate_B_Residual(data=data,B_In=B_In,B_Ex=B_Ex)

    # Now group them by 20min slice and Plot the data
    Fig_path = 'Result_pic'
    Plot_B_Residual_Group(B_residual,Time_Range=20,path=Fig_path)
    # End


def Plot_B_Residual_Group(B_residual,Time_Range=20,path=''):

    grouped_data = B_residual.groupby(pd.Grouper(freq=f'{Time_Range}T'))

    filename = f'{path}/FACs_Check'
    os.makedirs(filename, exist_ok=True)

    for time, group in grouped_data:

        plt.figure(figsize=(13, 8))

        plt.subplot(4, 2, 1)
        plt.title(f'Residual B Field (nT) \n {time} - {time + Juno_Mag_MakeData_Function.min_1 * Time_Range}')
        component = 'Bx'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.xticks([])

        plt.subplot(4, 2, 3)
        component = 'By'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.xticks([])

        plt.subplot(4, 2, 5)
        component = 'Bz'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.subplot(4, 2, 2)
        plt.title(f'Residual B Field (nT) \n {time} - {time + Juno_Mag_MakeData_Function.min_1 * Time_Range}')
        component = 'Br'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.xticks([])

        plt.subplot(4, 2, 4)
        component = 'Btheta'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.xticks([])

        plt.subplot(4, 2, 6)
        component = 'Bphi'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        plt.subplot(4, 2, (7, 8))
        component = 'Btotal'
        plt.plot(group.index, group[component])
        plt.ylabel(f'$\delta {component}$')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xlabel('Time')

        plt.tight_layout()

        plt.savefig(filename+f'/B_Residual_{Time_Range}Range_{time}.jpg',dpi=300)
        # plt.show()
        plt.close()

main()