# Exercise 5.4
# Author: Felix Cahyadi
# Date: 16.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Define the function for acceleration
def acceleration(x, G = 1, M = 10, L = 2):

    r = norm(x,2)
    ddx = -G*M*(x/((r**2)*np.sqrt(r**2 + (L**2/4))))

    return ddx

# Velocity verlet integration
def vel_verlet(accel, t, x0, v0):
    
    # define delta t
    deltat = t[1] - t[0]

    # Initiate arrays
    x_arr = np.zeros((len(x0), len(t)))
    v_arr = np.zeros((len(v0), len(t)))
    a_arr = np.zeros((len(x0), len(t)))

    # Initial values
    x_arr[:,0] = x0
    v_arr[:,0] = v0
    a_arr[:,0] = accel(x0)

    # Solve using verlet integration 
    for i in range(1, len(t)):
        x_arr[:,i] = x_arr[:,i-1] + v_arr[:,i-1]*deltat + 0.5*a_arr[:,i-1]*(deltat**2)
        a_arr[:,i] = accel(x_arr[:,i])
        v_arr[:,i] = v_arr[:,i-1] + 0.5*(a_arr[:,i-1]+a_arr[:,i])*deltat

    return x_arr, v_arr, a_arr

# Run the integration
t = np.linspace(0,10, 10000)
x0 = np.array([1,0])
v0 = np.array([0,1])

x_sol, v_sol, a_sol = vel_verlet(acceleration, t, x0, v0)

# Plot the solution
fig_traj, ax_traj = plt.subplots(figsize = (8,8))
ax_traj.plot(x_sol[0,:], x_sol[1,:])
ax_traj.set_title("Particle trajectory")
ax_traj.grid(True)


# Plot the values of position, velocity, and acceleration w.r.t. time
fig, ax = plt.subplots(nrows=3, ncols=1, figsize = (9,11))

ax[0].plot(t, x_sol[0,:], label = 'x component')
ax[0].plot(t, x_sol[1,:], label = 'y component')
ax[0].grid(True)
ax[0].set_xlabel("time")
ax[0].set_ylabel("position")
ax[0].legend()

ax[1].plot(t, v_sol[0,:], label = 'x component')
ax[1].plot(t, v_sol[1,:], label = 'y component')
ax[1].grid(True)
ax[1].set_xlabel("time")
ax[1].set_ylabel("velocity")
ax[1].legend()

ax[2].plot(t, a_sol[0,:], label = 'x component')
ax[2].plot(t, a_sol[1,:], label = 'y component')
ax[2].grid(True)
ax[2].set_xlabel("time")
ax[2].set_ylabel("acceleration")
ax[2].legend()

plt.show()
    
