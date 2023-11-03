# Final exam problem 2
# Author: Felix Cahyadi
# Date: 03.11.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
from scipy.integrate import solve_ivp, solve_bvp, simpson
from scipy.fft import fft
from numpy.linalg import norm
import fysa1130


# Create a function that returns the derivatives
def fun(x,y,k = 2):
    """This is a function that calculates dy/dx = y' and dy'/dx = -k*exp(y) and returns them in form of (2,1) array

    Args:
        x (float): The value of x where we want to evaluate the function
        y (ND array): Array containing the value of y and y' where we want to evaluate the function

    Returns:
        derivatives: dy/dx and dy'/dx
    """
    return np.array([y[1], -k*np.exp(y[0])])

# Create a function that defines the boundary condition
def bc(ya, yb):
    return np.array([ya[0], yb[0]])

x = np.linspace(0,1,100)

# Get the solutions using different guesses, we need to do this because Bratu's problem has two solutions
y1 = np.zeros((2, len(x)))
y2 = np.zeros((2, len(x)))
y2[0] = 2

sol_1 = solve_bvp(fun, bc, x, y1)
sol_2 = solve_bvp(fun, bc, x, y2)


# Plotting stuff

fig, ax = plt.subplots(figsize = (9,5))
ax.plot(sol_1.x, sol_1.y[0], label = "First solution")
ax.plot(sol_2.x, sol_2.y[0], label = "Second solution")
ax.grid(True)
ax.set_xlabel("$x$", fontsize = 16)
ax.set_ylabel("$y$", fontsize = 16)
ax.set_xlim([sol_1.x[0], sol_1.x[-1]])
ax.set_ylim([0, np.max(sol_2.y[0])+0.1])
ax.legend()

plt.show()