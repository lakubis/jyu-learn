# Exercise 3.2
# Author: Felix Cahyadi
# Date: 02.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from gnewton import gnewton

# Firstly, we are going to create the function that calculates the value of the LHS - RHS

def nonlin(x, ret_np = True):
    """the function the calculates the value of the LHS - RHS

    Args:
        x (list): The x and y value
        ret_np (bool, optional): return NumPy array if True. Defaults to True.

    Returns:
        sol: The value of the LHS - RHS
    """
    sol = []

    sol.append(((x[0] - 2)**2) + (x[1]**2) - 4)
    sol.append((x[0]**2) + ((x[1] - 3)**2) - 4)

    if ret_np:
        return np.array(sol)
    else:
        return sol

# We can then do some contour plot to identify some roots
x_arr = np.linspace(-6,6,1000)
y_arr = np.linspace(-6,6,1000)

X,Y = np.meshgrid(x_arr,y_arr)
Z1,Z2 = nonlin([X,Y], ret_np=False) # The value of the first and the second equation on different coordinates

fig, ax = plt.subplots(figsize = (5,5))
ax.contour(X,Y,np.abs(Z1),levels = [0.005],colors = ['r']) # We plot the absolute value so that we can detect values near zero without caring for the sign
ax.contour(X,Y,np.abs(Z2),levels = [0.005],colors = ['b'])
ax.grid(True)

print("From the plot, we acquired the values where the two equations are close to zero, and the two equations coincides near (0,1) and (2,2), this means we might have solutions there")
print("We are going to use (0,1) and (2,2) to guess the solution")

# Find the solutions using gnewton.py
guess1 = np.array([0,1])
guess2 = np.array([2,2])
sol1 = gnewton(nonlin, guess1)
sol2 = gnewton(nonlin, guess2)

ax.plot(sol1[0],sol1[1], 'g*')
ax.plot(sol2[0],sol2[1], 'g*')


print("The coordinates of the first solution: ", sol1)
print("The coordinates of the second solution: ", sol2)
print("The solutions are plotted using the green stars")
plt.show()