# Final exam problem 4
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

# Create a function that calculates the derivatives
def diff_y(x, y):
    dddy = 4*y[1]*np.sqrt(x+(y[0]**2))
    return np.array([y[1],y[2], dddy])

# Define the initial values
time_span = np.array([0,1])
y_init = np.array([2,1,0.1])

# Solve the problem using solve_ivp
solution = solve_ivp(diff_y, t_span= time_span, y0=y_init, method = 'RK45',max_step = 0.001)
x_array = solution.t
y_array = solution.y

print(f"We got the solution that y(1) = {y_array[0,-1]:.3f}")

# Plot the solution

fig, ax = plt.subplots(figsize = (9,5))
ax.plot(x_array, y_array[0], label = "$y$")
ax.grid(True)
ax.set_xlabel("$x$", fontsize = 16)
#ax.set_ylabel("")
ax.set_xlim([x_array[0], x_array[-1]])
#ax.set_ylim([])
ax.legend()

plt.show()