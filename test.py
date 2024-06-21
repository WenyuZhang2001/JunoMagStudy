import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import lpmn,factorial

# Nmax = 10
# theta_val = np.pi/4
#
#
# def schmidt_semi_normalization(n, m):
#     return ((-1) ** m) * np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))
#
#
# for n in range(1,Nmax + 1):
#     for m in range(n + 1):
#         P, dP = lpmn(m, n, np.cos(theta_val))
#         N_lm = schmidt_semi_normalization(n, m)
#         Snm = P[m,n]*N_lm
#         print(f'm = {m}  n = {n}')
#         print(f'P = {P[m,n]}')
#         print(f'N_ln = {N_lm}')
#         print(f'Snm = {Snm}')
#         print('='*50)

# print(True or True or True)

# a = np.array([1,2,3])
# b = np.array([4,5,6])
# c = np.array([7,8,9])
# df = pd.DataFrame()
# df['a']  = a
# df['b'] = b
# df['c'] = c
# print(np.vstack((a,b,c)))
# print(np.hstack((df['a'],df['b'],df['c'])).T.reshape(-1))

# theta_val = np.pi/2
# theta_val = 7/360*np.pi
# theta_val = 1e-10
# n = 1
# m = 1
# print(np.cos(theta_val))
# P1, dP1 = lpmn(m, n, np.cos(theta_val))
# P2, dP2 = lpmn(m, n, np.cos(theta_val)-1e-10)
# dP3 = (P1-P2)/2/1e-5
# print(f'P = {P1[m,n]}, dP = {dP1[m,n]}')
# print(dP3[m,n])

import Spherical_Harmonic_InversionModel_Functions

# gnm_hnm = Spherical_Harmonic_InversionModel_Functions.read_gnm_hnm_data(method='SVD', Nmax=1, path='Spherical_Harmonic_Model/First20_Orbit_Model')
#
# print(gnm_hnm)

# A = np.array(range(20))
#
# print(A[2::2])

# import spiceypy as spice
#
# spice.tkvrsn('TOOLKIT')
# print(spice.tkvrsn('TOOLKIT'))

# print(np.linspace(0, 24, 25))
# print(np.arange(0, 24, 1))
#
# Rc_Lambda_Dic = {'0.88': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
#                      '0.85': [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]}
#
#   # Gets the first key
# num_values = len(Rc_Lambda_Dic[next(iter(Rc_Lambda_Dic))])
# print(num_values)
#
# print('_'+str(0.1))

A = np.array([1,2,3])
B = np.array([4,5,6])
C = np.vstack((A,B)).T.reshape(-1)
print(C)
print(list(range(1,3)))
