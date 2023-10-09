# Exercise 4.3
# Author: Felix Cahyadi
# Date: 09.10.2023

# Import libraries
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the function that we want to minimize
print("Here, we will set V0 = 1.0")
def V(r,*args, V0 = 1.0, sigma = 1.0):
    """ The function that calculates the potential

    Args:
        r (float): The distance from the atomic center
        V0 (float, optional): V0 constant. Defaults to 1.0.
        sigma (float, optional): sigma constant. Defaults to 1.0

    Returns:
        potential : The potential
    """
    return V0*(((sigma/r)**6) - np.exp(-r/sigma))

# I think this problem would be easier to solve if we plot it
fig, ax = plt.subplots(figsize = (6,6))
r_arr = np.linspace(0.8,4,100)
V_arr = V(r_arr)
ax.plot(r_arr,V_arr)

result = minimize(V, 1.5)
r_opt = result['x'][0] # The optimized r coordinate

print(f"The optimized distance is r = {r_opt}, with V(r) = {V(r_opt)}")

# Plot the optimized result in the graph
ax.plot(r_opt, V(r_opt),'r*')
ax.set_ylabel("$V(r)$", fontsize = 16)
ax.set_xlabel("$r$", fontsize = 16)
ax.set_xlim([0.8,4])
ax.grid(True)
plt.show()