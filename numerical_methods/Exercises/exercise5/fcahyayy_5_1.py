# Exercise 5.1
# Author: Felix Cahyadi
# Date: 16.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Define the function for the acceleration
def ddx(x):
    """ This is the function that calculates the acceleration of the object

    Args:
        x (NDarray): The coordinate of the object

    Returns:
        acc: The acceleration vector
    """
    acc = -x/norm(x,2)**3 # to calculate the magnitude of the vector
    return acc

# Define the function for the implicit Euler method
def imp_euler(func, t, x0, dx0):
    """ Function for the implicit Euler method

    Args:
        func (function): The equation for d^2x/dt^2
        t (NDarray): Array containing timesteps
        x0 (NDarray): Initial position
        dx0 (NDarray): Initial velocity

    Returns:
        x, dx: position and velocity array
    """
    deltat = t[1]-t[0]

    x = np.zeros((len(x0), len(t))) # Create arrays to contain x
    dx = np.zeros((len(dx0), len(t))) # Create arrays to contain dx/dt
    x[:,0] = x0 # Initialize the position
    dx[:,0] = dx0 # Initialize the velocity

    for i in range(1,len(t)):
        dx[:,i] = dx[:,i-1] + deltat*func(x[:,i-1])
        x[:,i] = x[:,i-1] + deltat*dx[:,i] # semi-implicit euler method use i instead of i-1
    
    return x,dx

# Initialize the position and velocity
t = np.linspace(0,100,10000)
x0 = np.array([1.0,0.0])
dx0 = np.array([0.5,-1.0])

x, dx = imp_euler(ddx, t, x0, dx0)

print("As we can see, the trajectory is a circle, with a little bit of translation from the origin.")

# Plot the result
fig, ax = plt.subplots(figsize = (6,6))
ax.plot(x[0,:], x[1,:])
ax.set_title("Trajectory")
ax.grid(True)

# Plot the position and velocities over time
fig2, ax2 = plt.subplots(nrows=2, ncols=1, figsize = (8,8))
ax2[0].plot(t,x[0,:], label = "x component of the position")
ax2[0].plot(t,x[1,:], label = "y component of the position")
ax2[0].set_xlabel("time")
ax2[0].grid(True)
ax2[0].legend()

ax2[1].plot(t,dx[0,:], label = "x component of the velocity")
ax2[1].plot(t,dx[1,:], label = "y component of the velocity")
ax2[1].set_xlabel("time")
ax2[1].grid(True)
ax2[1].legend()
plt.show()