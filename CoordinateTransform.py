#!/usr/bin/env python
# coding: utf-8
from scipy.spatial.transform import Rotation as R
# In[53]:

import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np


# In[54]:


def SysIIItoJM_transform(vec):
    '''
    vec: the vec to transform form System III to Jupiter Magnetic  
    '''
    SysIIItoJM = Rotation.from_euler('zxy',[69.2,9.5,0],degrees=True)
    return SysIIItoJM.apply(vec)


# In[55]:


def JMtoSysIII_transform(vec):
    '''
    vec: the vec to transform form Jupiter Magnetic to System III  
    '''
    SysIIItoJM = Rotation.from_euler('zxy',[69.2,9.5,0],degrees=True)
    JMtoSysIII = SysIIItoJM.inv()
    return JMtoSysIII.apply(vec)


# In[1]:


def CartesiantoSpherical(x,y,z):
    
    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    
    return r,np.degrees(theta),np.degrees(phi)


# In[ ]:


def CartesiantoSpherical_Bfield(x,y,z,Bx,By,Bz):
    
    r,theta,phi = CartesiantoSpherical(x,y,z)
    theta = theta*2*np.pi/360
    phi = phi*2*np.pi/360
    
    Br = (Bx*np.cos(phi)+By*np.sin(phi))*np.sin(theta)+Bz*np.cos(theta)
    Btheta = (Bx*np.cos(phi)+By*np.sin(phi))*np.cos(theta)-Bz*np.sin(theta)
    Bphi = -Bx*np.sin(phi)+By*np.cos(phi)
    
    return Br,Btheta,Bphi

def SphericaltoCartesian_Bfield(r,theta,phi,Br,Btheta,Bphi):
    '''

    :param r: distance to Jupiter Center
    :param theta: Co latitude, in Degree
    :param phi: Longitude, in degree
    :param Br:
    :param Btheta:
    :param Bphi:
    :return:
    '''
    theta = theta * 2 * np.pi / 360
    phi = phi * 2 * np.pi / 360

    Bx = (Br*np.sin(theta)+Btheta*np.cos(theta))*np.cos(phi)-Bphi*np.sin(phi)
    By = (Br*np.sin(theta)+Btheta*np.cos(theta))*np.sin(phi)+Bphi*np.cos(phi)
    Bz = Br*np.cos(theta)-Btheta*np.sin(theta)

    return Bx,By,Bz

def SphericaltoCartesian(r,theta,phi):

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    return x,y,z
def calculate_rotation_matrix(row,Coord='Cartesian'):
    if Coord=='Cartesian':
        v1 = np.array([row['Bx'], row['By'], row['Bz']])
        v2 = np.array([row['Bx_ss'], row['By_ss'], row['Bz_ss']])
    elif Coord == 'Spherical':
        v1 = np.array([row['Br'], row['Btheta'], row['Bphi']])
        v2 = np.array([row['Br_ss'], row['Btheta_ss'], row['Bphi_ss']])

    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    rotation = R.align_vectors([v2_normalized], [v1_normalized])[0]
    return rotation.as_matrix()
def apply_rotation(row, bx, by, bz, Coord='Cartesian'):
    return pd.Series(row[f'rotation_matrix_{Coord}'].dot([bx, by, bz]))

def SysIIItoSS_Bfield(data,Bfield):

    # doing the Rotation matrix calculation PC to SS
    data['rotation_matrix_Cartesian'] = data.apply(calculate_rotation_matrix, axis=1, args=('Cartesian',))
    data['rotation_matrix_Spherical'] = data.apply(calculate_rotation_matrix, axis=1, args=('Spherical',))

    component_list = ['LocalTime', 'r', 'Latitude_ss', 'X_ss', 'Y_ss', 'Z_ss']
    for component in component_list:
        Bfield[component] = data[component]

        # Join df1 and df2 to apply rotations correctly
    df_combined = Bfield.join(data[['rotation_matrix_Cartesian', 'rotation_matrix_Spherical']])

    # Applying rotation to Bx, By, Bz
    Bfield[['Bx_ss', 'By_ss', 'Bz_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Bx'], row['By'], row['Bz'], Coord='Cartesian'), axis=1)
    # Applying rotation to Br, Btheta, Bphi
    Bfield[['Br_ss', 'Btheta_ss', 'Bphi_ss']] = df_combined.apply(
        lambda row: apply_rotation(row, row['Br'], row['Btheta'], row['Bphi'], Coord='Spherical'), axis=1)

    return Bfield