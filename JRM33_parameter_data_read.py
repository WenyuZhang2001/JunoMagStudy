import pandas as pd
import numpy as np

param = pd.read_csv('Result_data/JRM33_parameters.csv')


Nmax = 30

num_coeffs = int((Nmax + 2) * Nmax)

g_h_param = np.zeros(num_coeffs)

for n in range(1,Nmax+1):
    for m in range(n+1):
        gnm_index = int((n + 2) * (n - 1) / 2 + m)
        g_param_index = int((n-1+2)*(n-1)+m)
        g_h_param[gnm_index] = param.iloc[g_param_index]

        if m==0:
            continue
        hnm_index = int((n + 2) * (n - 1) / 2 - (n - 1) + m - 1 + (Nmax + 3) * Nmax / 2)
        h_param_index = int((n-1+2)*(n-1)+m+n)
        g_h_param[hnm_index] = param.iloc[h_param_index]

np.save('Result_data/JRM33_Coefficient.npy',g_h_param)

print(num_coeffs/2)
print(g_h_param[:int(num_coeffs/2)])
print(g_h_param[int((Nmax + 3) * Nmax / 2):])


