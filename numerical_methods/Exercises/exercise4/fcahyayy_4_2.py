# Exercise 4.2
# Author: Felix Cahyadi
# Date: 09.10.2023

# Import libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# Define the equations
def springs(x):
    """ The function that we use to calculate the potentials of the spring

    Args:
        x (list): a list containing the value of (x1, x2, y3, y4)

    Returns:
        potential energy: the sum of the potential energy of the springs
    """

    # Unpack values
    x1, x2, y3, y4 = x

    # Define the constants
    # lengths
    d = 0.1
    H = 1
    W = 1

    # spring constants
    k1 = 2
    k2 = 3
    k3 = 9
    k4 = 3
    k5 = 1

    # Calculate the energy
    E = []
    E.append(0.5*k1*(x1**2 + d**2))
    E.append(0.5*k2*((x2-x1)**2 + d**2))
    E.append(0.5*k3*((W-x2)**2 + y3**2))
    E.append(0.5*k4*(d**2 + (y4-y3)**2))
    E.append(0.5*k5*(d**2 + (-H-y4)**2))

    return np.sum(E)

# Define size
d = 0.1
H = 1
W = 1

# define the boundaries
bound = []
bound.append([0,W-d])
bound.append([0,W])
bound.append([-H,0])
bound.append([-H,-d])

# Initial guess
init_x = [0.1,0.2,-0.5,-0.5]

# Minimize stuff
result = minimize(springs,x0=init_x,bounds=bound)
print(f"This is the value of the potential energy: {result['fun']}")
coords = result['x']
x_arr = np.array([0,coords[0],coords[1], W, W-d, W]) # Collect the x values
y_arr = np.array([0,-d,0,coords[2],coords[3],-H]) # Collect the y values

print("These are the coordinates of the points:")
for i in range(1,len(x_arr)-1):
    print(f"P{i} = ({x_arr[i]},{y_arr[i]})")
print("Here, the origin is located on the top left.")

# plot
fig, ax = plt.subplots(figsize = (6,6))
ax.set_ylabel('y', fontsize = 16)
ax.set_xlabel('x', fontsize = 16)

# Plot the lines
ax.plot([0,W,W],[0,0,-H], linewidth = 2, color = 'black')
ax.plot([0,W-d,W-d],[-d,-d,-H], linewidth = 2, color = 'black')

# Plot the coordinates
ax.plot(x_arr,y_arr,'b', marker = 'o')
plt.show()