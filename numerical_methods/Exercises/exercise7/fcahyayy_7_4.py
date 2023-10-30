# Exercise 7.4
# Author: Felix Cahyadi
# Date: 30.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import fsolve

# Define the function that returns the derivatives
def wave_eq(t, states, E, m = 9.1e-31, hbar = 6.626e-34/(2*np.pi)):
    """ This is a function that calculates the value of the derivatives

    Args:
        t (float): time
        states (NDarray): array containing psi and phi
        E (float): The energy of the particle
        m (float, optional): Electron's mass. Defaults to 9.1e-31.
        hbar (float, optional): reduced planck's constant. Defaults to 6.626e-34/(2*np.pi).

    Returns:
        NDarray: The derivative of phi and psi
    """
    psi = states[1]
    phi = states[0]
    dpsi = phi
    dphi = ((2*m)/(hbar**2))*(-E)*psi

    return np.array([dphi,dpsi])

# Solve the states using rk4 algorithm, turn it into shooting method
def Energy_fit(E, initial_states, L, ret_sol = False):
    """ This is a function that calculates the wave function of the particle, or \psi(L) depending on the value of ret_sol

    Args:
        E (float): The energy of the system
        initial_states (NDarray): The initial state of psi and phi
        L (float): The length of the potential well
        ret_sol (bool, optional): If True, return the trajectory, if False, return only the potential. Defaults to False.

    Returns:
        solution: Will be an array containing the trajectory if ret_sol = True, will be \psi(L) if ret_sol = False.
    """
    sol = solve_ivp(wave_eq, (0, L), initial_states, method= 'RK45', args=([E]), max_step = 2e-14, vectorized= True)

    if ret_sol:
        return sol
    else:
        return sol.y[0][-1]

# Setting some initial values
L = 5.29177e-11
state0 = np.array([0.0,3.2746e-6]) # This value is acquired from the integration of the wave function below
guess_E = 100*1.602e-19


sol_E = fsolve(Energy_fit, guess_E, args = (state0, L))
print(f"The acquired energy is {sol_E[0]/1.602e-19} eV")

sol_traj = Energy_fit(sol_E, state0, L, ret_sol=True)

psi = sol_traj.y[0]
x = sol_traj.t

# Integrate the wave function
int_wave = simpson(np.square(psi), x)
print("The integral of the wave function $\int |\psi|^2 dx$ = ", int_wave)

# Visualization
fig, ax = plt.subplots(figsize = (9,5))
ax.plot(x, psi)
ax.grid(True)
ax.set_xlim((x[0],x[-1]))
ax.set_ylabel('\psi', fontsize = 16)
ax.set_xlabel('x', fontsize = 16)
plt.show()