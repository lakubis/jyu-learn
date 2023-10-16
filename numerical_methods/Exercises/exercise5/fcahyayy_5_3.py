# Exercise 5.3
# Author: Felix Cahyadi
# Date: 16.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4

# Define the differential equation
def dydx(x,y):
    
    return np.array([y[1],y[2],y[3],4*y[2]*np.sqrt(1-y[0]**2)])

# Run the RK algorithm
x_0 = 0
x_1 = 1
y_0 = np.array([0.1,0.1,0.1,0.1])
num_step = 100
h = (x_1 - x_0)/num_step

x_sol, y_sol = rk4(dydx=dydx, x0=x_0, x1 = x_1, y = y_0, h = h)

print(f"These are the trajectories, the final coordinates are \ny = {y_sol[-1,0]}, \ny' = {y_sol[-1,1]}, \ny'' = {y_sol[-1,2]}, \ny''' = {y_sol[-1,3]}")

# Plot the trajectories
fig, ax = plt.subplots(nrows=4, ncols=1, figsize = (9,16))

ax[0].plot(x_sol, y_sol[:,0])
ax[0].grid(True)
ax[0].set_xlabel('$x$', fontsize = 16)
ax[0].set_ylabel("$y$", fontsize = 16)

ax[1].plot(x_sol, y_sol[:,1])
ax[1].grid(True)
ax[1].set_xlabel('$x$', fontsize = 16)
ax[1].set_ylabel("$y'$", fontsize = 16)

ax[2].plot(x_sol, y_sol[:,2])
ax[2].grid(True)
ax[2].set_xlabel('$x$', fontsize = 16)
ax[2].set_ylabel("$y''$", fontsize = 16)

ax[3].plot(x_sol, y_sol[:,3])
ax[3].grid(True)
ax[3].set_xlabel('$x$', fontsize = 16)
ax[3].set_ylabel("$y'''$", fontsize = 16)

plt.show()

