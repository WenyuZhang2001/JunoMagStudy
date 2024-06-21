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
import Juno_Mag_MakeData_Function


# In[ ]:


def Plot_Juno_Position(data,pj=99,Savefig=True,ShowPlot=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return

    Time_start = data.index.min()
    Time_end = data.index.max()

    plt.figure(figsize=(15,15))

    plt.subplot(2,4,(1,2))
    component = 'r'
    plt.scatter(data[component].index,data[component],label = f'r')
    plt.ylabel(f'r $R_{{J}}$')
    plt.title(f'Juno Orbit r  \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2,4,(3,4))
    component = 'r'
    plt.scatter(data['Longitude'],data['Latitude'],label = f'r')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.title(f'Juno Orbit Latitude Longitude  \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.legend()
    plt.grid(True)
    
    
    ax = plt.subplot(2,4,(5,6),projection='3d')
    ax.scatter(data['Longitude'],data['DDay'],data['Latitude'].to_numpy(),label = 'Position')
    
    ax.set_yticks(data['DDay'].iloc[range(0,len(data),int(len(data)/5))],
               data['DDay'].iloc[range(0,len(data),int(len(data)/5))].index.hour)
    
    ax.set_xlim([0,360])
    ax.set_zlim([-90,90])
    ax.set_xlabel('Longitude')
    ax.set_zlabel('Latitude')
    ax.set_ylabel('Time (hrs)')
    plt.grid(True)
    
    plt.legend()
    ax.view_init(10,45,vertical_axis='z')
    
    ax = plt.subplot(2,4,(7),projection='3d')
    ax.scatter(data['Longitude'],data['DDay'],data['Latitude'].to_numpy(),label = 'Position')
    
    ax.set_yticks(data['DDay'].iloc[range(0,len(data),int(len(data)/5))],
               data['DDay'].iloc[range(0,len(data),int(len(data)/5))].index.hour)
    
    ax.set_xlim([0,360])
    ax.set_zlim([-90,90])
    # ax.set_xlabel('Longitude')
    ax.set_zlabel('Latitude')
    ax.set_ylabel('Time (hrs)')
    ax.set_xticks([])
    plt.grid(True)
    
    plt.legend()
    ax.view_init(0,0,vertical_axis='z')
    
    ax = plt.subplot(2,4,8,projection='3d')
    ax.scatter(data['Longitude'],data['DDay'],data['Latitude'].to_numpy(),label = 'Position')
    
    ax.set_yticks(data['DDay'].iloc[range(0,len(data),int(len(data)/5))],
               data['DDay'].iloc[range(0,len(data),int(len(data)/5))].index.hour)
    
    ax.set_xlim([0,360])
    ax.set_zlim([-90,90])
    ax.set_xlabel('Longitude')
    # ax.set_zlabel('Latitude')
    ax.set_zticks([])
    ax.set_ylabel('Time (hrs)')
    plt.grid(True)
    
    plt.legend()
    ax.view_init(90,180,vertical_axis='z')
    
    # plt.tight_layout()
    if Savefig:
        plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/r_{Time_start}.jpg',dpi=200)
    if ShowPlot:
        plt.show()
    else:
        plt.close()


# In[ ]:


def Plot_Bfeild(data,B_In,B_Ex,pj=99,Coordinate='Sys3',Savefig=True,Model='jrm33',ShowPlot=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return

    Time_start = data.index.min()
    Time_end = data.index.max()

    if Coordinate=='Sys3':
        plt.figure(figsize=(15,8))
    
        plt.subplot(4,1,1)
        plt.title(f'B Field Data and Model \n {Time_start}-{Time_end}')
        component = 'Bx'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.subplot(4,1,2)
        component = 'By'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.subplot(4,1,3)
        component = 'Bz'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.subplot(4,1,4)
        component = 'Btotal'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/Data_Ex+{Model}_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()
    elif Coordinate=='Spherical':
        plt.figure(figsize=(15,8))

        plt.subplot(4,1,1)
        plt.title(f'B Field Data and Model \n {Time_start}-{Time_end}')
        component = 'Br'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.subplot(4,1,2)
        component = 'Btheta'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.subplot(4,1,3)
        component = 'Bphi'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.subplot(4,1,4)
        component = 'Btotal'
        plt.scatter(B_Ex[component].index,B_Ex[component]+B_In[component],label='Model B_Ex+B_In')
        plt.scatter(data[component].index,data[component],label = 'data')
        plt.ylabel(component+'(nT)')
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/Data_Ex+{Model}_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()


# In[ ]:


def Plot_Delta_Bfield(data,B_In,B_Ex,pj=99,Coordinate='Sys3',
                      MinusBex=True,Savefig=True,Percentage_ylim=10,Model='jrm33',ShowPlot=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return

    Time_start = data.index.min()
    Time_end = data.index.max()

    if Coordinate=='Sys3' and (not MinusBex):
        plt.figure(figsize=(20,15))
    
        plt.subplot(4,2,1)
        plt.title(f'$\delta$B_component  \n {Time_start}-{Time_end}\n Orbit{pj:0>2d}')
        component = 'Bx'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,3)
        component = 'By'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,5)
        component = 'Bz'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,7)
        plt.title(f'$\delta$Btotal  \n {Time_start}-{Time_end}\n Orbit{pj:0>2d}')
        component = 'Btotal'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,2)
        plt.title(f'$\delta$B_component  \n {Time_start}-{Time_end}')
        component = 'Bx'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(4,2,4)
        component = 'By'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(4,2,6)
        component = 'Bz'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(4,2,8)
        plt.title(f'$\delta$Btotal  \n {Time_start}-{Time_end}')
        component = 'Btotal'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/Data-{Model}_Ex_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()
    elif Coordinate == 'Spherical' and (not MinusBex):
        plt.figure(figsize=(20,15))

        plt.subplot(4,2,1)
        plt.title(f'$\delta$B_component  \n {Time_start}-{Time_end}\n Orbit{pj:0>2d}')
        component = 'Br'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,3)
        component = 'Btheta'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,5)
        component = 'Bphi'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,7)
        plt.title(f'$\delta$Btotal  \n {Time_start}-{Time_end}\n Orbit{pj:0>2d}')
        component = 'Btotal'
        plt.scatter(B_Ex[component].index,B_Ex[component],label='Model B_Ex')
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,2)
        plt.title(f'$\delta$B_component  \n {Time_start}-{Time_end}')
        component = 'Br'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(4,2,4)
        component = 'Btheta'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(4,2,6)
        component = 'Bphi'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.subplot(4,2,8)
        plt.title(f'$\delta$Btotal  \n {Time_start}-{Time_end}')
        component = 'Btotal'
        plt.scatter(B_Ex[component].index,np.abs(B_Ex[component]),label='Model B_Ex')
        plt.scatter(data[component].index,np.abs(data[component]-B_In[component]),label = f'$\delta$ {component} (data-{Model})')
        plt.ylabel(f'$log_{{10}}$($\delta$'+component+') (nT)')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/Data-{Model}_Ex_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()
    elif Coordinate=='Sys3' and  MinusBex:
        plt.figure(figsize=(20,15))

        plt.subplot(4,2,1)
        plt.title(f'$\delta$B_component  \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
        component = 'Bx'
        plt.scatter(data[component].index,data[component]-B_In[component]-B_Ex[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,2)
        plt.title(r'$\frac{\delta B}{B}$')
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.subplot(4,2,3)
        component = 'By'
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,4)
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        
        plt.subplot(4,2,5)
        component = 'Bz'
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,6)
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btoal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.subplot(4,2,7)
        plt.title(f'$\delta$Btotal  \n {Time_start}-{Time_end}')
        component = 'Btotal'
        plt.scatter(data[component].index,data[component]-B_In[component]-B_Ex[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,8)
        plt.title(r'$\frac{\delta B}{B}$')
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/Data-{Model}-Ex|Btotal_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()
    elif Coordinate=='Spherical' and  MinusBex:
        plt.figure(figsize=(20,15))

        plt.subplot(4,2,1)
        plt.title(f'$\delta$B_component  \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
        component = 'Br'
        plt.scatter(data[component].index,data[component]-B_In[component]-B_Ex[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,2)
        plt.title(r'$\frac{\delta B}{B}$')
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.subplot(4,2,3)
        component = 'Btheta'
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,4)
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        
        plt.subplot(4,2,5)
        component = 'Bphi'
        plt.scatter(data[component].index,data[component]-B_In[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,6)
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btoal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.subplot(4,2,7)
        plt.title(f'$\delta$Btotal  \n {Time_start}-{Time_end}')
        component = 'Btotal'
        plt.scatter(data[component].index,data[component]-B_In[component]-B_Ex[component],label = f'$\delta$ {component} (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(4,2,8)
        plt.title(r'$\frac{\delta B}{B}$')
        plt.scatter(data[component].index,(data[component]-B_In[component]-B_Ex[component])/data['Btotal']*100,
                    label=f'$\delta$ {component}/Btotal')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/Data-{Model}-Ex|Btotal_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()


# In[ ]:


def Plot_Juno_Footprint(Juno_MAG_FP,pj=99,B_component='Br',ShowPlot=True,Savefig=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return
    plt.figure(figsize=(25,15))

    Time_start = Juno_MAG_FP.index.min()
    Time_end = Juno_MAG_FP.index.max()

    ax1 = plt.subplot(4,2,(1,2))
    plt.title(f'Juno Magnetic Field Line FootPrint North & South \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.scatter(Juno_MAG_FP['North_FP_Long'],Juno_MAG_FP['North_FP_Lat'],c=Juno_MAG_FP['r'],cmap='cool',label = 'North_FP Model')
    plt.scatter(Juno_MAG_FP['South_FP_Long'],Juno_MAG_FP['South_FP_Lat'],c=Juno_MAG_FP['r'],cmap='cool',label = 'South_FP Model')
    Plot_Jupiter_Surface_B(ax1,B_component=B_component)
    plt.ylim([-90,90])
    plt.xlim([0,360])
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.grid(True)
    cbar = plt.colorbar(pad=0.01,fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.legend()
    
    ax2 = plt.subplot(4,2,3)
    plt.title('Juno Magnetic Field Line FootPrint North')
    plt.scatter(Juno_MAG_FP['North_FP_Long'],Juno_MAG_FP['North_FP_Lat'],c=Juno_MAG_FP['r'],cmap='cool',label = 'North_FP Model')
    Plot_Jupiter_Surface_B(ax2,B_component=B_component)
    plt.ylim([0,90])
    plt.xlim([0,360])
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.grid(True)
    cbar = plt.colorbar(pad=0.01,fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.legend()
    
    ax3 = plt.subplot(4,2,4)
    plt.title('Juno Magnetic Field Line FootPrint South')
    plt.scatter(Juno_MAG_FP['South_FP_Long'],Juno_MAG_FP['South_FP_Lat'],c=Juno_MAG_FP['r'],cmap='cool',label = 'South_FP Model')
    Plot_Jupiter_Surface_B(ax3,B_component=B_component)
    plt.ylim([-90,0])
    plt.xlim([0,360])
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    plt.grid(True)
    cbar = plt.colorbar(pad=0.01,fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.legend()
    
    ax4 = plt.subplot(4,2,(5,7),projection='polar')
    plt.title('North Pole Azimuthal')
    ax4.scatter(Juno_MAG_FP['North_FP_Long']/360*2*np.pi,Juno_MAG_FP['North_FP_ArcLen'],c=Juno_MAG_FP['r'],
                cmap='cool',label = 'North_FP Model')
    ax4.set_rmax(np.pi/2)
    ax4.set_rticks([np.pi/2,np.pi/3,np.pi/6,0],labels=['0','30','60','Pole'])
    ax4.set_rlabel_position(30)
    plt.legend()
    Plot_Jupiter_Surface_B_PoleAzimuthal(ax4,Direction='North')
    # cbar = plt.colorbar()
    # cbar.set_label(f'Distance r ($R_{{J}}$)')
    ax4.grid(True)
    
    ax5 = plt.subplot(4,2,(6,8),projection='polar')
    plt.title('South Pole Azimuthal')
    ax5.scatter(Juno_MAG_FP['South_FP_Long']/360*2*np.pi,Juno_MAG_FP['South_FP_ArcLen'],c=Juno_MAG_FP['r'],
                cmap='cool',label = 'South_FP Model')
    Plot_Jupiter_Surface_B_PoleAzimuthal(ax5, Direction='South')
    ax5.set_rmax(np.pi/2)
    ax5.set_rticks([np.pi/2,np.pi/3,np.pi/6,0],labels=['0','30','60','Pole'])
    ax5.set_rlabel_position(30)
    ax5.grid(True)
    # cbar = plt.colorbar()
    # cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.legend()
    
    plt.tight_layout()
    if Savefig:
        plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/Juno_FP/Juno_FP_plot_{Time_start}.jpg',dpi=200)
    if ShowPlot:
        plt.show()
    else:
        plt.close()


# In[1]:


def Plot_Delta_Bfield_Btotal(data,B_In,B_Ex,pj=99,Coordinate='Sys3',
                      Savefig=True,Percentage_ylim=10,Model='jrm33',ShowPlot=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return

    Time_start = data.index.min()
    Time_end = data.index.max()

    if Coordinate=='Sys3':
        # Calculate
        delta_Bx = np.abs(data['Bx']-B_In['Bx']-B_Ex['Bx'])
        delta_By = np.abs(data['By']-B_In['By']-B_Ex['By'])
        delta_Bz = np.abs(data['Bz']-B_In['Bz']-B_Ex['Bz'])
        delta_Btotal = np.abs(data['Btotal']-B_In['Btotal']-B_Ex['Btotal'])
        Btotal_delta = np.sqrt(delta_Bx**2+delta_By**2+delta_Bz**2)

        # Plot
        plt.figure(figsize=(20,15))
        
        plt.subplot(3,1,1)
        plt.title(f'$\delta$B_total  \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
        component = 'Btotal'
        plt.scatter(data[component].index,delta_Btotal,label = f'$\delta$ |{component}| (data-{Model}-Ex)')
        plt.scatter(data[component].index,Btotal_delta,label = f'|$\delta$ {component}| (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()

        plt.subplot(3,1,2)
        component = 'Btotal'
        plt.plot(data[component].index,delta_Btotal-Btotal_delta,
                 label = f'$\delta$ |{component}|-|$\delta$ {component}| (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(3,1,3)
        component = 'Btotal'
        plt.plot(data[component].index,np.abs(delta_Btotal-Btotal_delta)/data['Btotal'],
                 label = f'$\delta$ |{component}|-|$\delta$ {component}|/Btotal (data-{Model}-Ex)')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/delta_Btotal|Btotal_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()
            
    elif Coordinate=='Spherical':
        # Calculate
        delta_Br = np.abs(data['Br']-B_In['Br']-B_Ex['Br'])
        delta_Btheta = np.abs(data['Btheta']-B_In['Btheta']-B_Ex['Btheta'])
        delta_Bphi = np.abs(data['Bphi']-B_In['Bphi']-B_Ex['Bphi'])
        delta_Btotal = np.abs(data['Btotal']-B_In['Btotal']-B_Ex['Btotal'])
        Btotal_delta = np.sqrt(delta_Br**2+delta_Btheta**2+delta_Bphi**2)

        # Plot
        plt.figure(figsize=(20,15))
        
        plt.subplot(3,1,1)
        plt.title(f'$\delta$B_total  \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
        component = 'Btotal'
        plt.scatter(data[component].index,delta_Btotal,label = f'$\delta$ |{component}| (data-{Model}-Ex)')
        plt.scatter(data[component].index,Btotal_delta,label = f'|$\delta$ {component}| (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()

        plt.subplot(3,1,2)
        component = 'Btotal'
        plt.plot(data[component].index,delta_Btotal-Btotal_delta,
                 label = f'$\delta$ |{component}|-|$\delta$ {component}| (data-{Model}-Ex)')
        plt.ylabel(f'$\delta$'+component+'(nT)')
        plt.legend()
        
        plt.subplot(3,1,3)
        component = 'Btotal'
        plt.plot(data[component].index,np.abs(delta_Btotal-Btotal_delta)/data['Btotal']*100,
                 label = f'$\delta$ |{component}|-|$\delta$ {component}|/Btotal (data-{Model}-Ex)')
        plt.ylabel(r'$\frac{\delta B}{Btotal}$%')
        if Percentage_ylim >0 :
            plt.ylim([-Percentage_ylim,Percentage_ylim])
        plt.legend()
        
        plt.tight_layout()
        if Savefig:
            plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/{Coordinate}/delta_Btotal|Btotal_{Time_start}.jpg',dpi=200)
        if ShowPlot:
            plt.show()
        else:
            plt.close()



def Plot_Jupiter_Surface_B(ax, B_component = 'Br',path='Result_data/Jupiter_Surface_B.csv'):
    # Load magnetic field data from CSV
    B = pd.read_csv(path)

    # Assuming the CSV file has columns: 'Longitude', 'Latitude', and 'Br'
    # and the shape of the data grid is correctly given as (100, 100)
    Shape = (100, 100)

    # Reshape and plot contour filled with Br values
    B_reshaped = B[B_component].to_numpy().reshape(Shape) / 1e5
    Longitude_reshaped = B['Longitude'].to_numpy().reshape(Shape)
    Latitude_reshaped = B['Latitude'].to_numpy().reshape(Shape)

    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)

    contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5,zorder=-1)

    # Create a colorbar for the contour plot
    fig = ax.figure  # Get the figure associated with the ax
    cbar_Surface = fig.colorbar(contourf_plot, ax=ax,pad=0.01,fraction=0.05)
    cbar_Surface.set_label('Br (Gauss)')


    # Plot contour lines over the filled contour and label them
    CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, alpha=0.7,zorder=-1)
    ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines


def Plot_Jupiter_Surface_B_PoleAzimuthal(ax, Direction='North', B_component='Br',
                                         path='Result_data/Jupiter_Surface_B.csv'):
    from scipy.interpolate import griddata
    # Load magnetic field data from CSV
    B = pd.read_csv(path)

    # Assuming the CSV file has columns: 'Longitude', 'Latitude', and 'Br'
    # and the shape of the data grid is correctly given as (100, 100)
    Shape = (100, 50)

    # Reshape and plot contour filled with Br values
    if Direction == 'North':
        B = B[B['Latitude'] >= 0]
        ArcLen = - B['Latitude'] * 2 * np.pi / 360 + np.pi / 2

        theta_grid, ArcLen_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi/2, 50))
        B_grid = griddata((B['Longitude'].to_numpy() / 360 * 2 * np.pi,ArcLen),B[B_component].to_numpy()/1e5,(theta_grid,ArcLen_grid))
        contourf_plot = ax.contourf(theta_grid, ArcLen_grid, B_grid, levels=20, cmap='jet', alpha=0.3)

        # Create a colorbar for the contour plot
        fig = ax.figure  # Get the figure associated with the ax
        cbar = fig.colorbar(contourf_plot, ax=ax)
        cbar.set_label('Br (Gauss)')

        # Plot contour lines over the filled contour and label them
        CS = ax.contour(theta_grid, ArcLen_grid, B_grid, levels=20, alpha=0.7)
        ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines
    elif Direction == 'South':
        B = B[B['Latitude'] <= 0]
        ArcLen = B['Latitude'] * 2 * np.pi / 360 + np.pi / 2

        theta_grid, ArcLen_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi / 2, 50))
        B_grid = griddata((B['Longitude'].to_numpy() / 360 * 2 * np.pi, ArcLen), B[B_component].to_numpy() / 1e5,
                          (theta_grid, ArcLen_grid))
        contourf_plot = ax.contourf(theta_grid, ArcLen_grid, B_grid, levels=20, cmap='jet', alpha=0.3)

        # Create a colorbar for the contour plot
        fig = ax.figure  # Get the figure associated with the ax
        cbar = fig.colorbar(contourf_plot, ax=ax)
        cbar.set_label('Br (Gauss)')

        # Plot contour lines over the filled contour and label them
        CS = ax.contour(theta_grid, ArcLen_grid, B_grid, levels=20, alpha=0.7)
        ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

def Plot_Juno_Footprint_thorughTime(Juno_MAG_FP,pj=99,ShowPlot=True,Savefig=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return
    plt.figure(figsize=(25,15))

    Time_start = Juno_MAG_FP.index.min()
    Time_end = Juno_MAG_FP.index.max()

    ax1 = plt.subplot(2,1,1)
    plt.title(f'Juno Magnetic Field Line FootPrint North\n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.scatter(Juno_MAG_FP.index, Juno_MAG_FP['North_FP_Lat'], c=Juno_MAG_FP['r'], cmap='cool',label='North_FP Model')
    plt.ylim([-90, 90])
    plt.ylabel('Latitude')
    plt.xlabel('Time')
    plt.grid(True)
    cbar = plt.colorbar(pad=0.01, fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    plt.title(f'Juno Magnetic Field Line FootPrint South\n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.scatter(Juno_MAG_FP.index, Juno_MAG_FP['South_FP_Lat'], c=Juno_MAG_FP['r'], cmap='cool', label='South_FP Model')
    plt.ylim([-90, 90])
    plt.ylabel('Latitude')
    plt.xlabel('Time')
    plt.grid(True)
    cbar = plt.colorbar(pad=0.01, fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.legend()

    plt.tight_layout()
    if Savefig:
        plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/Juno_FP/Juno_FP_plot_{Time_start}_throughTime.jpg', dpi=200)
    if ShowPlot:
        plt.show()
    else:
        plt.close()

def Plot_Juno_Footprint_Anomaly(Juno_MAG_FP,pj=99,Extmodel ='Con2020',MaxLen=5000,B_component='Br',ShowPlot=True,Savefig=True,path=''):
    if pj == 99:
        print('Wrong PJ input!')
        return

    Time_start = Juno_MAG_FP.index.min()
    Time_end = Juno_MAG_FP.index.max()

    plt.figure(figsize=(25,15))
    # The North Anomaly
    North_Anomaly = Juno_MAG_FP.loc[Juno_MAG_FP['North_FP_Lat'] < 0 ]
    North_Anomaly['r'] = np.ones(len(North_Anomaly))
    North_Anomaly['theta'] = - North_Anomaly['North_FP_Lat'] + 90

    South_Anomaly = Juno_MAG_FP.loc[Juno_MAG_FP['South_FP_Lat'] > 0]
    South_Anomaly['r'] = np.ones(len(South_Anomaly))
    South_Anomaly['theta'] = - South_Anomaly['South_FP_Lat'] + 90

    X,Y,Z = CoordinateTransform.SphericaltoCartesian(South_Anomaly['r'],South_Anomaly['theta'],South_Anomaly['North_FP_Long'])
    South_Anomaly['X'] = X ; South_Anomaly['Y'] = Y; South_Anomaly['Z'] = Z

    X, Y, Z = CoordinateTransform.SphericaltoCartesian(North_Anomaly['r'], North_Anomaly['theta'],
                                                       North_Anomaly['North_FP_Long'])
    North_Anomaly['X'] = X
    North_Anomaly['Y'] = Y
    North_Anomaly['Z'] = Z

    X, Y, Z = CoordinateTransform.SphericaltoCartesian(South_Anomaly['r'], South_Anomaly['theta'],
                                                       South_Anomaly['South_FP_Long'])
    South_Anomaly['X'] = X
    South_Anomaly['Y'] = Y
    South_Anomaly['Z'] = Z

    North_Anomaly_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(North_Anomaly,Extmodel=Extmodel,maxLen=MaxLen)
    South_Anomaly_FP = Juno_Mag_MakeData_Function.FootPrintCalculate(South_Anomaly,Extmodel=Extmodel,maxLen=MaxLen)

    ax1 = plt.subplot(2, 1, 1)
    plt.title(f'Juno Magnetic Field Line FootPrint North Anomaly \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.scatter(North_Anomaly['North_FP_Long'], North_Anomaly['North_FP_Lat'], c=North_Anomaly['r'], cmap='rainbow',marker='*',s=200,
                label='North_Anomaly')
    cbar = plt.colorbar(pad=0.01, fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.scatter(North_Anomaly_FP['North_FP_Long'], North_Anomaly_FP['North_FP_Lat'], label='North_FP')
    plt.scatter(North_Anomaly_FP['South_FP_Long'], North_Anomaly_FP['South_FP_Lat'], label='South_FP')
    Plot_Jupiter_Surface_B(ax1, B_component=B_component)
    plt.ylim([-90, 90])
    plt.xlim([0, 360])
    plt.ylabel('Latitude')
    plt.xlabel(f'Longitude \n Total North Anomaly Numbers:{len(North_Anomaly)}')
    plt.grid(True)
    for i, txt in enumerate(North_Anomaly_FP.index):
        plt.plot([North_Anomaly_FP['North_FP_Long'].iloc[i],North_Anomaly['North_FP_Long'].iloc[i],North_Anomaly_FP['South_FP_Long'].iloc[i]],
                 [North_Anomaly_FP['North_FP_Lat'].iloc[i],North_Anomaly['North_FP_Lat'].iloc[i],North_Anomaly_FP['South_FP_Lat'].iloc[i]],
                 c='white')
    plt.legend()

    ax2 = plt.subplot(2, 1, 2)
    plt.title(f'Juno Magnetic Field Line FootPrint South Anomaly \n {Time_start}-{Time_end} \n Orbit{pj:0>2d}')
    plt.scatter(South_Anomaly['South_FP_Long'], South_Anomaly['South_FP_Lat'], c=South_Anomaly['r'], cmap='rainbow',
                marker='*', s=200,
                label='South_Anomaly')
    cbar = plt.colorbar(pad=0.01, fraction=0.1)
    cbar.set_label(f'Distance r ($R_{{J}}$)')
    plt.scatter(South_Anomaly_FP['North_FP_Long'], South_Anomaly_FP['North_FP_Lat'], label='North_FP')
    plt.scatter(South_Anomaly_FP['South_FP_Long'], South_Anomaly_FP['South_FP_Lat'], label='South_FP')
    Plot_Jupiter_Surface_B(ax2, B_component=B_component)
    plt.ylim([-90, 90])
    plt.xlim([0, 360])
    plt.ylabel('Latitude')
    plt.xlabel(f'Longitude \n Total South Anomaly Numbers:{len(South_Anomaly)}')
    plt.grid(True)
    for i, txt in enumerate(South_Anomaly_FP.index):
        plt.plot([South_Anomaly_FP['North_FP_Long'].iloc[i],South_Anomaly['South_FP_Long'].iloc[i],South_Anomaly_FP['South_FP_Long'].iloc[i]],
                 [South_Anomaly_FP['North_FP_Lat'].iloc[i],South_Anomaly['South_FP_Lat'].iloc[i],South_Anomaly_FP['South_FP_Lat'].iloc[i]],
                 c='white')

    plt.legend()

    plt.tight_layout()
    if Savefig:
        plt.savefig(f'{path}/EachPeriJovian/Juno_Orbit_{pj:0>2d}/Juno_FP/Juno_FP_Anomaly_plot_{Time_start}_.jpg', dpi=200)
    if ShowPlot:
        plt.show()
    else:
        plt.close()