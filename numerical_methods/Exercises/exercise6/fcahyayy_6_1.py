# Exercise 6.1
# Author: Felix Cahyadi
# Date: 23.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Make a function that solve the differential equation y'(x) + x y(x) = A
def solve_diffeq(y_arr, x_arr, y_init, A = 10, max_iter = 100):
    """This is the function that calculates the value of y'(x) + xy(x) - A

    Args:
        y_arr (NDarray): Array containing ys
        x_arr (NDarray): Array containing xs
        y_init (float): Initial value of y
        A (int, optional): The constant A in the differential equation. Defaults to 10.
        max_iter (int, optional): The maximum number of iteration that can happen. Defaults to 100.

    Returns:
        F(x,y,dy): The value of y'(x) + xy(x) - A
    """
    h = (x_arr[-1]-x_arr[0])/(len(x_arr)-1)

    # Create dy
    dy_arr = np.zeros_like(y_arr)
    dy_arr[0] = (y_arr[1] - y_arr[0])/h
    dy_arr[-1] = (y_arr[-1] - y_arr[-2])/h
    dy_arr[1:-1] = (y_arr[2:] - y_arr[0:-2])/(2*h)

    # Value from the differential equation
    val_arr = dy_arr + x_arr*y_arr - A

    # Fix the boundary condition
    val_arr[0] = y_arr[0] - y_init # This will push fsolve to fix the initial condition

    return val_arr

# Initial value
N = 200
x = np.linspace(0,10,N) # Initialize the x
y = np.zeros(N)

y_sol = fsolve(solve_diffeq, y, args=(x,0))


fig, ax = plt.subplots(figsize = (9,5))
ax.plot(x,y_sol)
ax.grid(True)
ax.set_xlim((min(x),max(x)))
ax.set_ylim((min(y_sol), max(y_sol) + 1))
ax.set_ylabel("$y$", fontsize = 16)
ax.set_xlabel("$x$", fontsize = 16)
ax.tick_params("both", labelsize = 12)

plt.show()