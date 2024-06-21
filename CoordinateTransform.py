#!/usr/bin/env python
# coding: utf-8

# In[53]:


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