# Exercise 3.3
# Author: Felix Cahyadi
# Date: 02.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Firstly, we are going to create a function that calculates the value of the LHS - RHS
def nonlin2(x, ret_np = True):
    """the function the calculates the value of the LHS - RHS

    Args:
        x (list): The x and y value
        ret_np (bool, optional): return NumPy array if True. Defaults to True.

    Returns:
        sol: The value of the LHS - RHS
    """
    sol = []
    sol.append(x[0]-x[1]-(2*np.sin(x[0]+x[1])))
    sol.append(x[1] + (x[0]**2) - 2)
    
    if ret_np:
        return np.array(sol)
    else:
        return sol
    

# We can then do some contour plot to identify some roots
x_arr = np.linspace(-5,5,1000)
y_arr = np.linspace(-5,5,1000)

X,Y = np.meshgrid(x_arr,y_arr)
Z1,Z2 = nonlin2([X,Y], ret_np=False) # The value of the first and the second equation on different coordinates

fig, ax = plt.subplots(figsize = (5,5))
ax.contour(X,Y,np.abs(Z1),levels = [0.005],colors = ['r']) # We plot the absolute value so that we can detect values near zero without caring for the sign
ax.contour(X,Y,np.abs(Z2),levels = [0.005],colors = ['b'])
ax.grid(True)

print("From the plot, we acquired the values where the two equations are close to zero, and the two equations coincides near (-2,-4), (-2,-1), (-1,0), and (2,0), this means we might have solutions there")
print("We are going to use (-2,-4), (-2,-1), (-1,0), and (2,0) to guess the solution")

# Find the solutions using fsolve
guess1 = np.array([-2,-4])
guess2 = np.array([-2,-1])
guess3 = np.array([-1,0])
guess4 = np.array([2,0])
sol1 = fsolve(nonlin2, guess1)
sol2 = fsolve(nonlin2, guess2)
sol3 = fsolve(nonlin2, guess3)
sol4 = fsolve(nonlin2, guess4)

ax.plot(sol1[0],sol1[1], 'g*')
ax.plot(sol2[0],sol2[1], 'g*')
ax.plot(sol3[0],sol3[1], 'g*')
ax.plot(sol4[0],sol4[1], 'g*')

print("The coordinates of the first solution: ", sol1)
print("The coordinates of the second solution: ", sol2)
print("The coordinates of the third solution: ", sol3)
print("The coordinates of the fourth solution: ", sol4)
print("The solutions are plotted using the green stars")
plt.show()