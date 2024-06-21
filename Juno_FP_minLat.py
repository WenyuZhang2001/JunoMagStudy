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


Juno_MAG_FP_North = pd.read_csv('Result_data/Juno_MAG_FP_MinLatitudeNorth.csv',index_col=0)
Juno_MAG_FP_South = pd.read_csv('Result_data/Juno_MAG_FP_MinLatitudeSouth.csv',index_col=0)

B_component = 'Br'



plt.figure(figsize=(25,20))

ax1 = plt.subplot(4,2,(1,2))
plt.title(f'Juno Magnetic Field Line FootPrint North & South')
plt.scatter(Juno_MAG_FP_North['North_FP_Long'],Juno_MAG_FP_North['North_FP_Lat'],c=Juno_MAG_FP_North['r'],cmap='cool',label = 'North_FP Model')
cbar_North = plt.colorbar(pad=0.01,fraction=0.05)
cbar_North.set_label(f'Distance r ($R_{{J}}$) North')
plt.scatter(Juno_MAG_FP_South['South_FP_Long'],Juno_MAG_FP_South['South_FP_Lat'],c=Juno_MAG_FP_South['r'],cmap='Wistia',label = 'South_FP Model')
cbar_South = plt.colorbar(pad=0.02,fraction=0.05)
cbar_South.set_label(f'Distance r ($R_{{J}}$) South')
MyPlot_Functions.Plot_Jupiter_Surface_B(ax1,B_component=B_component)
plt.ylim([-90,90])
plt.xlim([0,360])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.grid(True)
for i, txt in enumerate(Juno_MAG_FP_North.index):
    plt.text(Juno_MAG_FP_North['North_FP_Long'].iloc[i], 
             Juno_MAG_FP_North['North_FP_Lat'].iloc[i], int(Juno_MAG_FP_North['PJ'].iloc[i]), fontsize=9)
for i, txt in enumerate(Juno_MAG_FP_South.index):
    plt.text(Juno_MAG_FP_South['South_FP_Long'].iloc[i], 
             Juno_MAG_FP_South['South_FP_Lat'].iloc[i], int(Juno_MAG_FP_South['PJ'].iloc[i]), fontsize=9)

plt.legend()

ax2 = plt.subplot(4,2,3)
plt.title('Juno Magnetic Field Line FootPrint North')
plt.scatter(Juno_MAG_FP_North['North_FP_Long'],Juno_MAG_FP_North['North_FP_Lat'],c=Juno_MAG_FP_North['r'],cmap='cool',label = 'North_FP Model')
plt.scatter(Juno_MAG_FP_North['South_FP_Long'],Juno_MAG_FP_North['South_FP_Lat'],c=Juno_MAG_FP_North['r'],cmap='cool')
MyPlot_Functions.Plot_Jupiter_Surface_B(ax2,B_component=B_component)
# plt.ylim([0,90])
plt.xlim([0,360])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.grid(True)
for i, txt in enumerate(Juno_MAG_FP_North.index):
    if Juno_MAG_FP_North['North_FP_Lat'].iloc[i] < 0:
        plt.plot([Juno_MAG_FP_North['North_FP_Long'].iloc[i], Juno_MAG_FP_North['South_FP_Long'].iloc[i]],
                 [Juno_MAG_FP_North['North_FP_Lat'].iloc[i], Juno_MAG_FP_North['South_FP_Lat'].iloc[i]], c='red')
        plt.text(Juno_MAG_FP_North['North_FP_Long'].iloc[i],
                 Juno_MAG_FP_North['North_FP_Lat'].iloc[i], int(Juno_MAG_FP_North['PJ'].iloc[i]), fontsize=9)
        plt.text(Juno_MAG_FP_North['South_FP_Long'].iloc[i],
                 Juno_MAG_FP_North['South_FP_Lat'].iloc[i], int(Juno_MAG_FP_North['PJ'].iloc[i]), fontsize=9)
        continue
    plt.plot([Juno_MAG_FP_North['North_FP_Long'].iloc[i],Juno_MAG_FP_North['South_FP_Long'].iloc[i]],
             [Juno_MAG_FP_North['North_FP_Lat'].iloc[i],Juno_MAG_FP_North['South_FP_Lat'].iloc[i]],c='white')
    plt.text(Juno_MAG_FP_North['North_FP_Long'].iloc[i], 
             Juno_MAG_FP_North['North_FP_Lat'].iloc[i], int(Juno_MAG_FP_North['PJ'].iloc[i]), fontsize=9)
    plt.text(Juno_MAG_FP_North['South_FP_Long'].iloc[i],
             Juno_MAG_FP_North['South_FP_Lat'].iloc[i], int(Juno_MAG_FP_North['PJ'].iloc[i]), fontsize=9)
cbar = plt.colorbar(pad=0.01,fraction=0.1)
cbar.set_label(f'Distance r ($R_{{J}}$)')
plt.legend()

ax3 = plt.subplot(4,2,4)
plt.title('Juno Magnetic Field Line FootPrint South')
plt.scatter(Juno_MAG_FP_South['South_FP_Long'],Juno_MAG_FP_South['South_FP_Lat'],c=Juno_MAG_FP_South['r'],cmap='cool',label = 'South_FP Model')
plt.scatter(Juno_MAG_FP_South['North_FP_Long'],Juno_MAG_FP_South['North_FP_Lat'],c=Juno_MAG_FP_South['r'],cmap='cool')
MyPlot_Functions.Plot_Jupiter_Surface_B(ax3,B_component=B_component)
# plt.ylim([-90,0])
plt.xlim([0,360])
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.grid(True)
for i, txt in enumerate(Juno_MAG_FP_South.index):
    if Juno_MAG_FP_South['South_FP_Lat'].iloc[i] > 0:
        plt.plot([Juno_MAG_FP_South['North_FP_Long'].iloc[i], Juno_MAG_FP_South['South_FP_Long'].iloc[i]],
                 [Juno_MAG_FP_South['North_FP_Lat'].iloc[i], Juno_MAG_FP_South['South_FP_Lat'].iloc[i]], c='red')
        plt.text(Juno_MAG_FP_South['South_FP_Long'].iloc[i],
                 Juno_MAG_FP_South['South_FP_Lat'].iloc[i], int(Juno_MAG_FP_South['PJ'].iloc[i]), fontsize=9)
        plt.text(Juno_MAG_FP_South['North_FP_Long'].iloc[i],
                 Juno_MAG_FP_South['North_FP_Lat'].iloc[i], int(Juno_MAG_FP_South['PJ'].iloc[i]), fontsize=9)
        continue
    plt.plot([Juno_MAG_FP_South['North_FP_Long'].iloc[i], Juno_MAG_FP_South['South_FP_Long'].iloc[i]],
             [Juno_MAG_FP_South['North_FP_Lat'].iloc[i], Juno_MAG_FP_South['South_FP_Lat'].iloc[i]], c='white')
    plt.text(Juno_MAG_FP_South['South_FP_Long'].iloc[i], 
             Juno_MAG_FP_South['South_FP_Lat'].iloc[i], int(Juno_MAG_FP_South['PJ'].iloc[i]), fontsize=9)
    plt.text(Juno_MAG_FP_South['North_FP_Long'].iloc[i],
             Juno_MAG_FP_South['North_FP_Lat'].iloc[i], int(Juno_MAG_FP_South['PJ'].iloc[i]), fontsize=9)
cbar = plt.colorbar(pad=0.01,fraction=0.1)
cbar.set_label(f'Distance r ($R_{{J}}$)')
plt.legend()

ax4 = plt.subplot(4,2,(5,7),projection='polar')
plt.title('North Pole Azimuthal')
ax4.scatter(Juno_MAG_FP_North['North_FP_Long']/360*2*np.pi,Juno_MAG_FP_North['North_FP_ArcLen'],c=Juno_MAG_FP_North['r'],
            cmap='cool',label = 'North_FP Model',s=200,marker='*',edgecolors='black')
MyPlot_Functions.Plot_Jupiter_Surface_B_PoleAzimuthal(ax=ax4)
ax4.set_rmax(np.pi/2)
ax4.set_rticks([np.pi/2,np.pi/3,np.pi/6,0],labels=['0','30','60','Pole'])
ax4.set_rlabel_position(30)
plt.legend()
ax4.grid(True)
for i, txt in enumerate(Juno_MAG_FP_North.index):
    if Juno_MAG_FP_North['North_FP_Lat'].iloc[i] < 0:
        continue
    plt.text(Juno_MAG_FP_North['North_FP_Long'].iloc[i]/360*2*np.pi, 
             Juno_MAG_FP_North['North_FP_ArcLen'].iloc[i], int(Juno_MAG_FP_North['PJ'].iloc[i]), fontsize=9)

ax5 = plt.subplot(4,2,(6,8),projection='polar')
plt.title('South Pole Azimuthal')
ax5.scatter(Juno_MAG_FP_South['South_FP_Long']/360*2*np.pi,Juno_MAG_FP_South['South_FP_ArcLen'],c=Juno_MAG_FP_South['r'],
            cmap='cool',label = 'South_FP Model',s=200,marker='*',edgecolors='black')
MyPlot_Functions.Plot_Jupiter_Surface_B_PoleAzimuthal(ax=ax5,Direction='South')
ax5.set_rmax(np.pi/2)
ax5.set_rticks([np.pi/2,np.pi/3,np.pi/6,0],labels=['0','30','60','Pole'])
ax5.set_rlabel_position(30)
ax5.grid(True)
plt.legend()
for i, txt in enumerate(Juno_MAG_FP_South.index):
    if Juno_MAG_FP_South['South_FP_Lat'].iloc[i] > 0:
        continue
    plt.text(Juno_MAG_FP_South['South_FP_Long'].iloc[i]/360*2*np.pi, 
             Juno_MAG_FP_South['South_FP_ArcLen'].iloc[i], int(Juno_MAG_FP_South['PJ'].iloc[i]), fontsize=9)

plt.tight_layout()
plt.savefig(f'Result_pic/Juno_FP_MinLat.jpg',dpi=200)
plt.show()


# In[ ]:




