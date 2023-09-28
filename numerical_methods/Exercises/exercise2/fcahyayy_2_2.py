# Exercise 2.2
# Author: Felix Cahyadi
# Date: 26.09.2023

# Import libraries
import numpy as np

# Create the function f(x)
f = lambda x: (x**3) - 2*np.pi*(x**2) - 9*x + 1

# Define function to calculate the derivative of a function
def diff(func, x, h = 1e-4):
    """Function to differentiate a mathematical function

    Args:
        func (function): The function that we want to evaluate
        x (float): The position where we want to evaluate the derivative
        h (float, optional): The step size. Defaults to 1e-4.

    Returns:
        float: The derivative of the function at point x
    """
    return (func(x+h)-func(x-h))/(2*h)

# Define function for Newton-Raphson method
def NewRaph(func, x_0, tol = 1e-8, max_iter = 100):
    """ This is a function to apply the Newton-Raphson method.

    Args:
        func (function): Function that we want to evaluate
        x_0 (float): Initial guess
        tol (float, optional): The tolerance that we want to use for zero-finding. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of iteration, so that it doesn't stuck in an infinite loop. Defaults to 100.

    Returns:
        float: The roots
    """
    x_prev = x_0
    count = 0
    while np.abs(x_prev)>tol and count<=max_iter:
        x_next = x_prev - (func(x_prev)/diff(func, x_prev))
        x_prev = x_next
        count += 1

    if count == max_iter:
        print('Reached the maximum number of iteration')

    return x_next

# Define function to find the solution between the range (xa,xb) and N partition
def find_roots(func, x_min, x_max, N = 1000):
    """Finding roots the brute force way using Newton-Raphson method

    Args:
        func (function): The function that we want to examine
        x_min (float): The lower limit of the search space
        x_max (float): The upper limit of the search space
        N (int): The number of partition of the search space, default: 1000

    Returns:
        NDarray: NumPy array containing the roots of the calculations
    """

    search_arr = np.linspace(x_min,x_max,N) # create the search space

    roots = np.array([],dtype=np.float64) # Create an empty array to contain the roots

    for s in search_arr: # Iterate through the search space
        root = NewRaph(func,s) # Calculate the root
        if not (root in roots): # If the root is new, append it to roots
            roots = np.append(roots, root)
            

    return roots

# Using self-made function to find the roots
man_roots = find_roots(f,-10,10,1000)

# Using numpy to find the roots
np_roots = np.roots([1,-2*np.pi,-9,1])

print(50*'-')
print("Newton-Raphson roots: ", man_roots)
print(50*'-')
print("NumPy roots: ", np_roots)
print(50*'-')
