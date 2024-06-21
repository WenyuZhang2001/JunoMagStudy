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


Coordinate = 'Spherical'

Juno_MAG_Max_delta_Btotal = pd.read_csv(f'Result_data/Juno_MAG_MaxDeltaBtotal_{Coordinate}.csv')
Juno_MAG_Max_Btotal_delta = pd.read_csv(f'Result_data/Juno_MAG_MaxBtotalDelta_{Coordinate}.csv')

Juno_FP_Max_delta_Btotal = Juno_Mag_MakeData_Function.FootPrintCalculate(Juno_MAG_Max_delta_Btotal)
Juno_FP_Max_Btotal_delta = Juno_Mag_MakeData_Function.FootPrintCalculate(Juno_MAG_Max_Btotal_delta)

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
plt.title(f'Juno MAG |$\delta$ Btotal| {Coordinate}')
plt.bar(Juno_MAG_Max_Btotal_delta['Longitude'],Juno_MAG_Max_Btotal_delta['Btotal_delta'],width=3)
plt.ylabel(f'|$\delta$ Btotal| nT')
plt.xlabel('Longitude')
for i, txt in enumerate(Juno_MAG_Max_Btotal_delta.index):
    plt.text(Juno_MAG_Max_Btotal_delta['Longitude'].iloc[i],
             Juno_MAG_Max_Btotal_delta['Btotal_delta'].iloc[i], int(Juno_MAG_Max_Btotal_delta['PJ'].iloc[i]), fontsize=9)
# plt.xticks(Juno_MAG_Max_Btotal_delta['Longitude'], Juno_MAG_Max_Btotal_delta['Longitude'].astype(int), rotation=45)
plt.grid()

plt.subplot(2,2,2)
plt.title(f'Juno MAG $\delta$ |Btotal| {Coordinate}')
plt.bar(Juno_MAG_Max_delta_Btotal['Longitude'],Juno_MAG_Max_delta_Btotal['delta_Btotal'],width=3)
plt.ylabel(f'$\delta$ |Btotal| nT')
plt.xlabel('Longitude')
for i, txt in enumerate(Juno_MAG_Max_delta_Btotal.index):
    plt.text(Juno_MAG_Max_delta_Btotal['Longitude'].iloc[i],
             Juno_MAG_Max_delta_Btotal['delta_Btotal'].iloc[i], int(Juno_MAG_Max_delta_Btotal['PJ'].iloc[i]), fontsize=9)
# plt.xticks(Juno_MAG_Max_delta_Btotal['Longitude'],Juno_MAG_Max_delta_Btotal['Longitude'].astype(int), rotation=45)
plt.grid()

ax3 = plt.subplot(2,2,3)
plt.title(f'Juno Magnetic Field Line FootPrint North & South')
plt.scatter(Juno_FP_Max_Btotal_delta['North_FP_Long'],Juno_FP_Max_Btotal_delta['North_FP_Lat'],label = 'North_FP Model')
plt.scatter(Juno_FP_Max_Btotal_delta['South_FP_Long'],Juno_FP_Max_Btotal_delta['South_FP_Lat'],label = 'South_FP Model')
plt.scatter(Juno_MAG_Max_Btotal_delta['Longitude'],Juno_MAG_Max_Btotal_delta['Latitude'],c = Juno_MAG_Max_Btotal_delta['delta_Btotal'],
            cmap ='cool', label='Juno Position',marker='*',s=200)
cbar_South = plt.colorbar(pad=0.1,fraction=0.1)
cbar_South.set_label(f'|$\delta$ Btotal| nT')
MyPlot_Functions.Plot_Jupiter_Surface_B(ax3,B_component='Br')
plt.ylim([-90,90])
plt.xlim([0,360])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.grid(True)
for i, txt in enumerate(Juno_MAG_Max_Btotal_delta.index):
    plt.text(Juno_MAG_Max_Btotal_delta['Longitude'].iloc[i],
             Juno_MAG_Max_Btotal_delta['Latitude'].iloc[i], int(Juno_MAG_Max_Btotal_delta['PJ'].iloc[i]), fontsize=9)
    plt.plot([Juno_FP_Max_Btotal_delta['North_FP_Long'].iloc[i],Juno_MAG_Max_Btotal_delta['Longitude'].iloc[i],Juno_FP_Max_Btotal_delta['South_FP_Long'].iloc[i]],
             [Juno_FP_Max_Btotal_delta['North_FP_Lat'].iloc[i],Juno_MAG_Max_Btotal_delta['Latitude'].iloc[i], Juno_FP_Max_Btotal_delta['South_FP_Lat'].iloc[i]],c='white')
plt.legend()

ax4 = plt.subplot(2,2,4)
plt.title(f'Juno Magnetic Field Line FootPrint North & South')
plt.scatter(Juno_FP_Max_delta_Btotal['North_FP_Long'],Juno_FP_Max_delta_Btotal['North_FP_Lat'],label = 'North_FP Model')
plt.scatter(Juno_FP_Max_delta_Btotal['South_FP_Long'],Juno_FP_Max_delta_Btotal['South_FP_Lat'],label = 'South_FP Model')
plt.scatter(Juno_MAG_Max_delta_Btotal['Longitude'],Juno_MAG_Max_delta_Btotal['Latitude'],c = Juno_MAG_Max_delta_Btotal['delta_Btotal'],
            cmap ='cool', label='Juno Position',marker='*',s=200)
cbar_South = plt.colorbar(pad=0.1,fraction=0.1)
cbar_South.set_label(f'$\delta$ |Btotal| nT')
MyPlot_Functions.Plot_Jupiter_Surface_B(ax4,B_component='Br')
plt.ylim([-90,90])
plt.xlim([0,360])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.grid(True)
for i, txt in enumerate(Juno_MAG_Max_delta_Btotal.index):
    plt.text(Juno_MAG_Max_delta_Btotal['Longitude'].iloc[i],
             Juno_MAG_Max_delta_Btotal['Latitude'].iloc[i], int(Juno_MAG_Max_delta_Btotal['PJ'].iloc[i]), fontsize=9)
    plt.plot([Juno_FP_Max_delta_Btotal['North_FP_Long'].iloc[i],Juno_MAG_Max_delta_Btotal['Longitude'].iloc[i],Juno_FP_Max_delta_Btotal['South_FP_Long'].iloc[i]],
             [Juno_FP_Max_delta_Btotal['North_FP_Lat'].iloc[i],Juno_MAG_Max_delta_Btotal['Latitude'].iloc[i],Juno_FP_Max_delta_Btotal['South_FP_Lat'].iloc[i]],c='white')

plt.legend()



plt.tight_layout()
plt.savefig(f'Result_pic/Juno_MAG_delta_Btotal_{Coordinate}_Longitude.jpg',dpi=200)
plt.show()