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
import CoordinateTransform
from scipy.special import lpmn,factorial
import joblib
import os


# make data point location
Nmax = 2
# Replace 'path_to_your_csv_file.csv' with the actual path to your CSV file
gnm_hnm_coeffi = np.genfromtxt(f'Spherical_Harmonic_Model/optimized_coefficients_gnm_hnm_Nmax{Nmax}.csv', delimiter=',')

# Assuming the first column contains 'gnm' and the second column contains 'hnm' coefficients
half_len = len(gnm_hnm_coeffi) // 2
gnm = gnm_hnm_coeffi[:half_len].reshape((Nmax + 1, Nmax + 1))
hnm = gnm_hnm_coeffi[half_len:].reshape((Nmax + 1, Nmax + 1))
def compute_magnetic_field_components(gnm, hnm, r, theta, phi):
    a = 1.0  # Reference radius for Jupiter, normalized to 1
    Nmax = len(gnm) - 1
    Br = np.zeros(len(theta))
    Btheta = np.zeros(len(theta))
    Bphi = np.zeros(len(theta))

    for idx, (r_val, theta_val, phi_val) in enumerate(zip(r, theta, phi)):
        cos_theta_val = np.cos(theta_val)
        P, dP = lpmn(Nmax, Nmax, cos_theta_val)

        for n in range(1, Nmax + 1):
            for m in range(n + 1):
                # Schmidt semi-normalization factor
                normalization_factor = np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))
                # Adjust for the Condon-Shortley phase
                Condon_Shortley_phase = (-1) ** m

                # Apply the normalization and phase to Pnm and its derivative
                Pnm = P[m, n] * normalization_factor * Condon_Shortley_phase
                dPnm = dP[m, n] * normalization_factor * Condon_Shortley_phase

                g_nm = gnm[n, m]
                h_nm = hnm[n, m]

                # Compute magnetic field components
                Br[idx] += (n + 1) * (a / r_val) ** (n + 2) * (
                            g_nm * np.cos(m * phi_val) + h_nm * np.sin(m * phi_val)) * Pnm
                Btheta[idx] += (a / r_val) ** (n + 2) * (
                            g_nm * np.cos(m * phi_val) + h_nm * np.sin(m * phi_val)) * dPnm * (-np.sin(theta_val))
                Bphi[idx] += (a / r_val) ** (n + 2) * m * (
                            g_nm * np.sin(m * phi_val) - h_nm * np.cos(m * phi_val)) * Pnm / np.sin(theta_val)

    return Br, Btheta, Bphi


Rj = 2
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
data['theta'] = theta/360*2*np.pi
data['phi'] = phi/360*2*np.pi
data['Longitude'] = Longitude.flatten()
data['Latitude'] = Latitude.flatten()
# B_In = Juno_Mag_MakeData_Function.MagneticField_Internal(data)
# data['Br'] = B_In['Br']
# data['Btheta'] = B_In['Btheta']
# data['Bphi'] = B_In['Bphi']
# data['Btotal'] = B_In['Btotal']
# data['Bx'] = B_In['Bx']
# data['By'] = B_In['By']
# data['Bz'] = B_In['Bz']

# data.to_csv('Result_data/Jupiter_Surface_B.csv')

plt.figure()

ax = plt.subplot(3,1,1)
plt.title('Jupiter Surface Br Model JRM33')

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

    contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5)

    # Create a colorbar for the contour plot
    fig = ax.figure  # Get the figure associated with the ax
    cbar = fig.colorbar(contourf_plot, ax=ax)
    cbar.set_label('Br (Gauss)')

    # Plot contour lines over the filled contour and label them
    CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=20, alpha=0.7)
    ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

Plot_Jupiter_Surface_B(ax)


def Plot_Jupiter_Surface_B_PoleAzimuthal(ax,Direction ='North', B_component = 'Br',path='Result_data/Jupiter_Surface_B.csv'):
    # Load magnetic field data from CSV
    B = pd.read_csv(path)

    # Assuming the CSV file has columns: 'Longitude', 'Latitude', and 'Br'
    # and the shape of the data grid is correctly given as (100, 100)
    Shape = (100, 50)

    # Reshape and plot contour filled with Br values
    if Direction == 'North':
        B = B[B['Latitude']>=0]
        ArcLen =  - B['Latitude']*2*np.pi/360 + np.pi/2
        B_reshaped = B[B_component].to_numpy().reshape(Shape) / 1e5
        Longitude_reshaped = B['Longitude'].to_numpy().reshape(Shape)/360*2*np.pi
        ArcLen_reshaped = ArcLen.to_numpy().reshape(Shape)

        contourf_plot = ax.contourf(Longitude_reshaped, ArcLen_reshaped, B_reshaped, levels=20, cmap='jet', alpha=0.5)

        # Create a colorbar for the contour plot
        fig = ax.figure  # Get the figure associated with the ax
        cbar = fig.colorbar(contourf_plot, ax=ax)
        cbar.set_label('Br (Gauss)')

        # Plot contour lines over the filled contour and label them
        CS = ax.contour(Longitude_reshaped, ArcLen_reshaped, B_reshaped, levels=20, alpha=0.7)
        ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines
    elif Direction == 'South':
        B = B[B['Latitude'] <= 0]
        ArcLen = B['Latitude'] * 2 * np.pi / 360 + np.pi / 2
        B_reshaped = B[B_component].to_numpy().reshape(Shape) / 1e5
        Longitude_reshaped = B['Longitude'].to_numpy().reshape(Shape)/360*2*np.pi
        ArcLen_reshaped = ArcLen.to_numpy().reshape(Shape)
        contourf_plot = ax.contourf(Longitude_reshaped, ArcLen_reshaped, B_reshaped, levels=20, cmap='jet', alpha=0.5)

        # Create a colorbar for the contour plot
        fig = ax.figure  # Get the figure associated with the ax
        cbar = fig.colorbar(contourf_plot, ax=ax)
        cbar.set_label('Br (Gauss)')

        # Plot contour lines over the filled contour and label them
        CS = ax.contour(Longitude_reshaped, ArcLen_reshaped, B_reshaped, levels=20, alpha=0.7)
        ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

ax = plt.subplot(3,1,2)
plt.title('Jupiter Surface Br Forward Model ')
# B_Model = calculate_Bfield(data,Nmax=14,path = 'Spherical_Harmonic_Model/2h_1sData',method='LSTSQ')
Shape = (100, 100)
# B_reshaped = B_Model['Br'].to_numpy().reshape(Shape) / 1e5
Br_pred, Btheta_pred, Bphi_pred = compute_magnetic_field_components(gnm,hnm,data['r'],data['theta'],data['phi'])
B_reshaped = Br_pred.reshape(Shape) / 1e5
Longitude_reshaped = data['Longitude'].to_numpy().reshape(Shape)
Latitude_reshaped = data['Latitude'].to_numpy().reshape(Shape)

min_value = -20
max_value = 20
levels = np.linspace(min_value, max_value, num=21)

contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5)

# Create a colorbar for the contour plot
fig = ax.figure  # Get the figure associated with the ax
cbar = fig.colorbar(contourf_plot, ax=ax)
cbar.set_label('Br (Gauss)')

# Plot contour lines over the filled contour and label them
CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=20, alpha=0.7)
ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

ax = plt.subplot(3,1,3)
Nmax = 3
plt.title(f'Jupiter Surface Br SVD Model Nmax = {Nmax}')
B_Model = calculate_Bfield(data,Nmax=Nmax,path = 'Spherical_Harmonic_Model/24h_60sData',method='SVD')
Shape = (100, 100)
B_reshaped = B_Model['Br'].to_numpy().reshape(Shape) / 1e5
Longitude_reshaped = data['Longitude'].to_numpy().reshape(Shape)
Latitude_reshaped = data['Latitude'].to_numpy().reshape(Shape)

min_value = -20
max_value = 20
levels = np.linspace(min_value, max_value, num=21)

contourf_plot = ax.contourf(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=levels, cmap='jet', alpha=0.5)

# Create a colorbar for the contour plot
fig = ax.figure  # Get the figure associated with the ax
cbar = fig.colorbar(contourf_plot, ax=ax)
cbar.set_label('Br (Gauss)')

# Plot contour lines over the filled contour and label them
CS = ax.contour(Longitude_reshaped, Latitude_reshaped, B_reshaped, levels=20, alpha=0.7)
ax.clabel(CS, fontsize=9, inline=True)  # Label contour lines

plt.show()