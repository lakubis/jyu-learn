# Exercise 5.2
# Author: Felix Cahyadi
# Date: 16.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from rk4 import rk4

# Define the differential equation
def dydx(x,y):
    """ Function that calculates the differential equation

    Args:
        x (float): The x
        y (NDarray): The ys

    Returns:
        dy: The derivatives of the ys w.r.t x
    """
    dy = (x**2) - 4*y
    return dy


# Run the RK algorithm
x_0 = 0
x_1 = 1
y_0 = np.array([1.0])
num_step = 20
h = (x_1 - x_0)/num_step

x_sol, y_sol = rk4(dydx, x_0, x_1, y_0, h)

# The analytical value of y(1)
analytical_y1 = 0.17399327517346125

# Function to find the number of steps needed
num_step_min = 10
num_step_max = 50

def find_num(dydx,x0,x1,y, theo_y1,num_step_min, num_step_max):
    """Function to 

    Args:
        dydx (function): Function that calculates the derivatives
        x0 (float): The starting point of x
        x1 (float): The end point of x
        y (ND_array): Initial condition of the ys
        theo_y1 (float): The theoretical value of y(1)
        num_step_min (int): The minimum number of steps
        num_step_max (int): The maximum number of steps

    Returns:
        i: The "Efficient" number of steps
    """

    for i in range(num_step_min, num_step_max):
        h = (x1 - x0)/i
        _,y_sol = rk4(dydx, x0, x1, y, h)

        if np.abs(y_sol[-1] - theo_y1)< 5e-7: # Our tolerance
            return i
    

# Finding the "Efficient" number of steps    
eff_num_step = find_num(dydx, x_0, x_1, y_0, analytical_y1, num_step_min, num_step_max)
h_eff = (x_1 - x_0)/eff_num_step

# Extract the solution at "Efficient" number of steps
_, ysol_eff = rk4(dydx, x_0, x_1, y_0, h_eff)
  
  

# Print y(1)
print(f"The value of y(1) from the simulation is {ysol_eff[-1][0]:.7f}\nThe difference with the analytical value is {analytical_y1 - ysol_eff[-1][0]:.6f} at {eff_num_step} steps.")

# Plot the result
fig, ax = plt.subplots(figsize = (9,5))
ax.plot(x_sol,y_sol)
ax.set_xlabel("x", fontsize = 16)
ax.set_ylabel("y", fontsize = 16)
ax.tick_params("both", labelsize = 16)
ax.grid(True)
plt.show()

