# Exercise 4.4
# Author: Felix Cahyadi
# Date: 09.10.2023

# Import libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the function that we want to minimize
def y_difference(x):
    """ Calculating the distance between two curves

    Args:
        x (float): the x coordinate

    Returns:
        y difference: the difference in y of both curves at x
    """
    
    y1 = np.sin(x**(1.55))
    y2 = 5*((x-np.e)**2) - 0.88

    return np.abs(y1-y2) # It's the absolute value because we are searching for the minimum distance

# Plot it so that everything is easier to solve
x_arr = np.linspace(1.5,4,100)
y_arr = y_difference(x_arr)

fig, ax = plt.subplots(figsize = (8,8))
ax.plot(x_arr, y_arr)

# Here, we know that x is not negative, otherwise the value x**1.55 would be a complex number. 
result = minimize(y_difference, 3)
x_opt = result['x'][0] # The optimized x coordinate
y_diff = result['fun']

print(f"The optimized x = {x_opt}, with |y_1 - y_2| = {y_diff}")

# Plot the optimized result in the graph
ax.plot(x_opt, y_diff, 'r*')
ax.set_ylabel("$|y_1 - y_2|$", fontsize = 16)
ax.set_xlabel("$x$", fontsize = 16)
ax.set_xlim([1.5,4])
ax.grid(True)

plt.show()