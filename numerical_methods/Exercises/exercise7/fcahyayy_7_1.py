# Exercise 7.1
# Author: Felix Cahyadi
# Date: 30.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define a function that gives the derivative
def derivative(x, y):
    ddy = (2+x)*y
    return ddy

# Define the function that calculates y'(1) given y'(0)
def trajectory(dy0, x, y0, ret_traj = False):
    """This is a function that calculates y'(1) given y'(0)

    Args:
        dy0 (float): The initial velocity
        x (NDarray): The array containing the values of x
        y0 (float): The initial y
        ret_traj (bool, optional): If true, return the trajectories, if false, return dy-5 for optimization by fsolve. Defaults to False.

    Returns:
        y, dy or dy[-1] - 5: Depends of the value of ret_traj, either return trajectory or only return last element of dy minus 5.
    """

    # Calculating h
    h = x[1] - x[0]

    # Calculate the trajectory
    y = np.zeros_like(x)
    y[0] = y0
    dy = np.zeros_like(x)
    dy[0] = dy0

    # Looping for the next y and dy
    for i in range(1,len(y)):
        dy[i] = dy[i-1] + h*derivative(x[i-1],y[i-1])
        y[i] = y[i-1] + h*dy[i-1]


    if ret_traj:
        return y, dy # Return the trajectories
    else:
        return dy[-1] - 5 # Because we want y'(1) = 5

# Define initial conditions
x = np.linspace(0,1,1000)
y0 = 0
dy_guess = 1

# Solve for the correct y'(0)
dy_sol = fsolve(trajectory, dy_guess,(x,y0))
print(f"The value of y'(0) is {dy_sol[0]}")

# Using the acquired y'(0) to get the y and y' trajectories
y_traj, dy_traj = trajectory(dy_sol[0], x, y0, ret_traj=True)

#Plot the result
fig,axs = plt.subplots(nrows=2, ncols=1, figsize = (9,10))
axs[0].plot(x,y_traj)
axs[0].grid(True)
axs[0].set_xlim((x[0],x[-1]))
axs[0].set_ylim((y_traj[0],y_traj[-1]))
axs[0].set_ylabel('$y$', fontsize = 16)
axs[0].set_xlabel('$x$', fontsize = 16)

axs[1].plot(x,dy_traj)
axs[1].grid(True)
axs[1].set_xlim((x[0],x[-1]))
axs[1].set_ylim((dy_traj[0],dy_traj[-1]))
axs[1].set_ylabel("$y'$", fontsize = 16)
axs[1].set_xlabel('$x$', fontsize = 16)

plt.show()
