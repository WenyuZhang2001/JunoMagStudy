import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lpmn,factorial
from scipy.optimize import minimize
import csv

Model = 'jrm33'
# Orbits and Doy time
data = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_Data.csv')
B_Ex = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_Ex.csv')
B_In = pd.read_csv('JunoFGMData/Processed_Data/Fist_20_Orbits_B_In.csv')

data['theta'] = data['theta']/360*2*np.pi
data['phi'] = data['phi']/360*2*np.pi

Br_obs = data['Br'] - B_Ex['Br']
Btheta_obs = data['Btheta'] - B_Ex['Btheta']
Bphi_obs = data['Bphi'] - B_Ex['Bphi']


# The Spherical Harmonic Model to Caluclate the B field
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

def objective_function(coeffs, r, theta, phi, Br_obs, Btheta_obs, Bphi_obs, Nmax):
    # Split coefficients into gnm and hnm
    half_len = len(coeffs) // 2
    gnm = coeffs[:half_len].reshape((Nmax + 1, Nmax + 1))
    hnm = coeffs[half_len:].reshape((Nmax + 1, Nmax + 1))

    # Compute predicted magnetic field components
    Br_pred, Btheta_pred, Bphi_pred = compute_magnetic_field_components(gnm, hnm, r, theta, phi)

    # Compute RMS error
    rms_error = np.sqrt(
        np.mean((Br_pred - Br_obs) ** 2 + (Btheta_pred - Btheta_obs) ** 2 + (Bphi_pred - Bphi_obs) ** 2))

    rms_error_toJRM = np.sqrt(
        np.mean((Br_pred - B_In['Br']) ** 2 + (Btheta_pred - B_In['Btheta']) ** 2 + (Bphi_pred - B_In['Bphi']) ** 2))
    RMS_error_hist.append(rms_error)
    RMS_error_toJRM_hist.append(rms_error_toJRM)
    return rms_error

def callback_function(current_coeffs):
    # This function is called after each iteration
    # print(f"Current coefficients: {current_coeffs}")
    print(f"Current RMS error: {RMS_error_hist[-1]}")

for Nmax in range(1,3):
    # The Order and Degree I use

    # initialized the gnm and hnm parameters
    initial_gnm_hnm =  np.zeros((2 * (Nmax + 1)**2, ))

    RMS_error_hist = []
    RMS_error_toJRM_hist = []
    result = minimize(objective_function, initial_gnm_hnm, args=(data['r'], data['theta'], data['phi'], Br_obs, Btheta_obs, Bphi_obs, Nmax),
                      method='SLSQP',callback=callback_function, options={'disp': True})  # Consider specifying bounds or other optimization options



    if result.success:
        fitted_coeffs = result.x
        print("Fitted coefficients:", fitted_coeffs)
    else:
        print("Optimization failed:", result.message)
        fitted_coeffs = result.x


    # Assuming 'coefficients' is a 1D NumPy array of your optimized coefficients
    with open(f'Spherical_Harmonic_Model/Forward_coefficients_gnm_hnm_Nmax{Nmax}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fitted_coeffs)

    with open(f'Spherical_Harmonic_Model/Forward_rms_errors_Nmax{Nmax}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'RMS Error','RMS_Error_toJRM'])  # Optional: write a header

        for i, (error_data, error_model) in enumerate(zip(RMS_error_hist, RMS_error_toJRM_hist)):
            writer.writerow([i, error_data,error_model])

    print(f'RMS errors have been saved to rms_errors_Nmax{Nmax}.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(RMS_error_hist, marker='o', linestyle='-', color='blue',label='RMS Error data')
    plt.plot(RMS_error_toJRM_hist, marker='*', linestyle='-', color='green',label=f'RMS Error {Model}')
    plt.title('RMS Error per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('RMS Error')
    plt.grid(True)
    plt.savefig(f'Spherical_Harmonic_Model/RMS_hist_pic/RMS_error_hist_Nmax{Nmax}.jpg',dpi=300)
    plt.close()