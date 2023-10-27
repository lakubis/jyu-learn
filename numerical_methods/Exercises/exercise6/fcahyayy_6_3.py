# Exercise 6.3
# Author: Felix Cahyadi
# Date: 23.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Create a function that returns dy and y
def diff_y(t,y,g = 9.81):
    """ Function that returns the derivatives of y

    Args:
        t (float): time
        y (NDarray): array containing y and its derivatives
        g (float, optional): Acceleration of gravity. Defaults to 9.81.

    Returns:
        dys: The derivatives of y
    """
    ddy = -g
    dy = y[-1]

    return np.array([dy, ddy])

# Create a function that returns final y given y0 using RK45
def propagate(v0, t_init, t_end, diff_y, y0):
    """ Function that calculates the final y using RK45, also makes it compatible with fsolve

    Args:
        v0 (float): The initial velocity
        t_init (float): The initial time
        t_end (float): The end time
        diff_y (function): The function that gives the derivative of y and v
        y0 (float): Initial value of y

    Returns:
        final y: The final value of y at t = 10 s
    """

    y_final = solve_ivp(diff_y, (t_init, t_end), np.array([y0, v0[0]]), method = 'RK45',max_step = 1e-1).y[0,-1]

    return y_final

# Set some initial conditions
t_init = 0
t_end = 10
v0 = 30 # This is the first guess
y0 = 0


# Solve for v0 using fsolve
v_sol = fsolve(propagate, v0, args=(t_init, t_end, diff_y, 0))

print(f"The initial velocity is: {v_sol[0]:.2f} m/s")
print(f"The value that we got from this method might be more accurate than the one that we have in 6.2.\nBecause in 6.2, I found the initial velocity using finite difference.\nThis method is way more time efficient compared to 6.2.")
