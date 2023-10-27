# Exercise 6.2
# Author: Felix Cahyadi
# Date: 23.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# Make a function that solves ddy/ddt = -g
def solve_grav(y_arr, y_init, y_end, h, tol = 1e-6, max_iter = 1e6):
    """This is the function that calculates y given ddy/ddt = -g and some boundary conditions

    Args:
        y_arr (NDarray): Array containing the values of y
        y_init (float): The first boundary of y
        y_end (float): The second boundary of y
        h (float): The step size
        tol (float, optional): The tolerance that we want to achieve. Defaults to 1e-6.
        max_iter (int, optional): The maximum iteration before getting out of the loop. Defaults to 100000.

    Returns:
        y_arr: The solution of y, also prints "Doesn't converge" if it doesn't converge.
    """
    # Define constant
    g = 9.81

    # Define the boundary condition
    y_arr[0] = y_init
    y_arr[-1] = y_end
    
    # Do the looping
    for i in range(int(max_iter)):
        #print(y_arr)
        norm_prev = norm(y_arr)
        y_left = y_arr[0:-2]
        y_right = y_arr[2:]
        y_arr[1:-1] = (y_right + y_left + (h**2)*g)/2
        norm_next = norm(y_arr)
        if abs((norm_next - norm_prev)/norm_next) < tol:
            print(f"It takes {i} iteration to converge.")
            return(y_arr)
        
    print("doesn't converge")
    return(y_arr)

# Initial value
N = 300
t, h = np.linspace(0,10,N, retstep= True) # Initialize the x
y = np.zeros(N)

y_sol = solve_grav(y, y_init=0, y_end = 0, h=h, tol=1e-9, max_iter=1e8)

fig, ax = plt.subplots(figsize = (9,5))
ax.plot(t,y_sol)
ax.grid(True)
ax.set_xlim((min(t),max(t)))
ax.set_ylim((min(y_sol), max(y_sol) + 5))
ax.set_ylabel("$y$ (m)", fontsize = 16)
ax.set_xlabel("$t$ (s)", fontsize = 16)
ax.tick_params("both", labelsize = 12)

g = 9.81
v0 = (y_sol[1] - y_sol[0])/h # Find the initial velocity using finite difference

print(f"The velocity is {v0} m/s")

plt.show()