import numpy as np

from scipy.special import lpmn,factorial
Nmax = 4

# P, dP = lpmn(Nmax, Nmax, np.cos(theta_val))

# print(P[0,3])
# print((5/2)*np.cos(theta_val)*((np.cos(theta_val))**2-9/15))
#
# print(P[2,3])
# print((15)**(0.5)/2*np.cos(theta_val)*(np.sin(theta_val)**2))
def schmidt_semi_normalization(n, m):
    return ((-1) ** m) * np.sqrt((2 - (m == 0)) * factorial(n - m) / factorial(n + m))

r_val = 1
phi_val = np.pi/4
theta_val = 89.9 /180 * np.pi
for n in range(1, Nmax + 1):
    for m in range(n + 1):
        P, dP = lpmn(m, n, np.cos(theta_val))
        N_lm = schmidt_semi_normalization(n, m)

        print(f'g[{n},{m}] = ',m * (r_val**(-n - 2)) * np.sin(m * phi_val)  * P[m, n] / np.sin(theta_val) * N_lm)

        if m == 0:
            continue
        print(f'h[{n},{m}] = ',m * (r_val**(-n - 2)) * np.sin(m * phi_val)  * P[m, n] / np.sin(theta_val) * N_lm)
