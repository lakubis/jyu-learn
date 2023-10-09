# Exercise 4.1
# Author: Felix Cahyadi
# Date: 09.10.2023

#Import libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Create the function that returns the three equation
def parabola(x):
    """A function that returns the sum of squares of the three equation

    Args:
        x (list): List containing v, theta, and h

    Returns:
        result: the sum of squares of the three equation
    """
    
    # Unpack the values
    v = x[0]
    theta = x[1] #in radian
    h = x[2]

    # Some constants
    g = 9.81
    t = 3.48

    # Return an array
    y = []
    # Append the first equation
    y.append((v*np.cos(theta)*t)-78.88) # From horizontal distance
    y.append(g*t - v*np.sin(theta) - v*np.cos(theta)*np.tan(np.deg2rad(38))) # From angle equation
    y.append(h + v*np.sin(theta)*t - 0.5*g*(t**2)) # From vertical distance

    return np.sum(np.square(y)) # sum the square of everything so that the minimization tends toward zero

# We use guess the initial vector as 22,0.1,1
result = minimize(parabola, [27,0.9,3], method='powell')
print(f"This is the value for the optimized function: {result['fun']}. \nIt's practically zero.\nWhich means we found the input values where our function produces zero.")
print(f"The optimized parameters:\n -Initial velocity: {result['x'][0]} m/s\n -Initial angle: {result['x'][1]} rad \n -Initial height = {result['x'][2]} m")

# To show that this is the right solution, we are going to plot the trajectory
def trajectory(v, theta, h, t, g = 9.81):
    x = v*np.cos(theta)*t # The x trajectory
    y = h + v*np.sin(theta)*t - 0.5*g*(t**2) # The y trajectory
    return x, y

# Create the time array to calculate the trajectory
t_arr = np.linspace(0,3.48, 100) # Plot the time from t = 0 to t = 3.48
x_arr, y_arr = trajectory(result['x'][0], result['x'][1], result['x'][2], t_arr)

# Plot the trajectory
fig, ax = plt.subplots(figsize = (9,5))
ax.plot(x_arr, y_arr)
ax.set_xlim((0,78.88))
ax.set_ylim((0,17))
ax.set_xlabel('x', fontsize = 16)
ax.set_ylabel('y', fontsize = 16)

# Draw the arc patch using the result that we have acquired, to show that we acquired the correct value
angle_patch = Arc((78.88,0),width=27, height=15, theta1 = 180-np.rad2deg(result['x'][1]), theta2 = 180)
ax.add_patch(angle_patch)

print("As we can see, using the values of the Initial velocity, angle, \nand height, we can acquire the trajectory that fits our requirements.")

plt.show()