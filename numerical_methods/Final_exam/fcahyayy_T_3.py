# Final exam problem 3
# Author: Felix Cahyadi
# Date: 03.11.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
from scipy.integrate import solve_ivp, solve_bvp, simpson
from scipy.fft import fft
from numpy.linalg import norm
import fysa1130

# This problem is a linear equations problem in form of Ax = b
# A contains the quantity of the items, x contains the price for each items, and b contains the total prices

# Create matrix A
A = np.array([[1.8, 1, 400], [0.8, 2, 400], [1.5, 3, 150]])

# Create matrix b
b = np.array([5.27, 4.77, 3.04])

# Solve for x using np.linalg.solve
x = np.linalg.solve(A,b)

# Amy's groceries
Amy = np.array([1,3,500])

# print the prices
print(f"The price for apples is EUR {x[0]:.2f} per kg")
print(f"The price for juice is EUR {x[1]:.2f} per liter")
print(f"The price for peanuts is EUR {x[2]:.2f} per gram")

# Print the solution
print(f"The total price for Amy's groceries is EUR {np.dot(Amy, x):.2f}")
# print(np.dot(np.array([1.5,3,150]),x))