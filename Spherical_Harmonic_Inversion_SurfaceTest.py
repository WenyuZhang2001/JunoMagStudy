import numpy as np
from scipy.linalg import lstsq
import pandas as pd
from scipy.special import lpmn,factorial
from sklearn.linear_model import Ridge
import joblib
import matplotlib.pyplot as plt
import os
import CoordinateTransform
import Juno_Mag_MakeData_Function
import Spherical_Harmonic_InversionModel_Functions
import MyPlot_Functions

# Make Surface meshgrid X, Y
Rj = 1
longitude = np.linspace(0,360,100)
latitude = np.linspace(-89,89,100)
Longitude, Latitude = np.meshgrid(longitude,latitude)
theta = 90 - Latitude
X = Rj*np.sin(theta/360*2*np.pi)*np.cos(Longitude/360*2*np.pi)
Y = Rj*np.sin(theta/360*2*np.pi)*np.sin(Longitude/360*2*np.pi)
Z = Rj*np.cos(theta/360*2*np.pi)
r , theta, phi = CoordinateTransform.CartesiantoSpherical(X.flatten(),Y.flatten(),Z.flatten())
data = pd.DataFrame(columns=['X','Y','Z'])
data['X'] = X.flatten()
data['Y'] = Y.flatten()
data['Z'] = Z.flatten()
data['r'] = r
data['theta'] = theta
data['phi'] = phi
data['Longitude'] = Longitude.flatten()
data['Latitude'] = Latitude.flatten()


Model = 'jrm33'
B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data,model=Model)


# path = 'Spherical_Harmonic_Model/Surface_Data'
path = 'Spherical_Harmonic_Model/2h_1sData'
# Make Dir
os.makedirs(path,exist_ok=True)

# Spherical_Harmonic_InversionModel_Functions.Model_Simulation(data,B_In,path=path,NMIN=1,NMAX=15)

Nmax = 20
B_Model_SVD = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,path=path,method = 'SVD',Nmax=Nmax)
B_Model_Ridge = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,path=path,method = 'Ridge',Nmax=Nmax)
B_Model_LSTSQ = Spherical_Harmonic_InversionModel_Functions.calculate_Bfield(data,path=path,method = 'LSTSQ',Nmax=Nmax)

def Plot_Model_Surface_Bfield(data,B_Model_SVD,B_Model_Ridge,B_Model_LSTSQ,component = 'Br',Nmax = 10):

    plt.figure(figsize=(8,13))

    # JRM33 Model
    ax = plt.subplot(4,1,1)
    plt.title('Jupiter Surface Br Model JRM33')
    MyPlot_Functions.Plot_Jupiter_Surface_B(ax,B_component=component)


    # SVD Model
    ax = plt.subplot(4, 1, 2)
    plt.title(f'Jupiter Surface Br SVD Model Nmax = {Nmax}')
    Shape = (100, 100)
    B_reshaped = B_Model_SVD['Br'].to_numpy().reshape(Shape) / 1e5
    Longitude_reshaped = data['Longitude'].to_numpy().reshape(Shape)
    Latitude_reshaped = data['Latitude'].to_numpy().reshape(Shape)

    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)

    contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5)

    # Create a colorbar for the contour plot
    fig = ax.figure  # Get the figure associated with the ax
    cbar = fig.colorbar(contourf_plot, ax=ax,pad=0.01,fraction=0.05)
    cbar.set_label('Br (Gauss)')
    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)
    # Plot contour lines over the filled contour and label them
    CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, alpha=0.7)
    ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

    # Ridge Model
    ax = plt.subplot(4, 1, 3)
    plt.title(f'Jupiter Surface Br Ridge Model Nmax = {Nmax}')
    Shape = (100, 100)
    B_reshaped = B_Model_Ridge['Br'].to_numpy().reshape(Shape) / 1e5
    Longitude_reshaped = data['Longitude'].to_numpy().reshape(Shape)
    Latitude_reshaped = data['Latitude'].to_numpy().reshape(Shape)

    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)

    contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5)

    # Create a colorbar for the contour plot
    fig = ax.figure  # Get the figure associated with the ax
    cbar = fig.colorbar(contourf_plot, ax=ax, pad=0.01, fraction=0.05)
    cbar.set_label('Br (Gauss)')
    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)
    # Plot contour lines over the filled contour and label them
    CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, alpha=0.7)
    ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

    # LSTSQ Model
    ax = plt.subplot(4, 1, 4)
    plt.title(f'Jupiter Surface Br LSTSQ Model Nmax = {Nmax}')
    Shape = (100, 100)
    B_reshaped = B_Model_LSTSQ['Br'].to_numpy().reshape(Shape) / 1e5
    Longitude_reshaped = data['Longitude'].to_numpy().reshape(Shape)
    Latitude_reshaped = data['Latitude'].to_numpy().reshape(Shape)

    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)

    contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5)

    # Create a colorbar for the contour plot
    fig = ax.figure  # Get the figure associated with the ax
    cbar = fig.colorbar(contourf_plot, ax=ax, pad=0.01, fraction=0.05)
    cbar.set_label('Br (Gauss)')
    min_value = -20
    max_value = 20
    levels = np.linspace(min_value, max_value, num=21)
    # Plot contour lines over the filled contour and label them
    CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, alpha=0.7)
    ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    # plt.savefig('Spherical_Harmonic_Model/Jupiter_Surface_Br.jpg',dpi=300)
    plt.show()


Plot_Model_Surface_Bfield(data,B_Model_SVD=B_Model_SVD,B_Model_Ridge=B_Model_Ridge,B_Model_LSTSQ=B_Model_LSTSQ,Nmax=Nmax)







