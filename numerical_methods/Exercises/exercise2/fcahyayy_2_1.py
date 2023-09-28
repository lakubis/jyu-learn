# Exercise 2.1
# Author: Felix Cahyadi
# Date: 26.09.2023

# Import libraries
import numpy as np
from numpy.linalg import solve

# We will turn the equation into matrix A and B.
# Make matrix A
A = np.zeros((10,10)) # Initialize it using the zeros matrices
A[0,0:3] = [7,-4,1]
A[1,0:4] = [-4,6,-4,1]
for i in range(2,8):
    A[i,i-2:i+3] = [1,-4,6,-4,1]
A[8,6:10] = [1,-4,6,-4]
A[9,7:10] = [1,-4,7]
print("This is the matrix A: \n", A)

# Make matrix B
B = np.ones(10)
print("This is the matrix B: ", B)

# Solve the linear equations using np.solve
sol = solve(A,B)
print("This is the solution: ", sol)

