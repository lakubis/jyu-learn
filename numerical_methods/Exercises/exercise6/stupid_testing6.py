# Exercise 6.4
# Author: Felix Cahyadi
# Date: 23.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# Create a function that will create two square shaped areas
def place_squares(grid, val, shift = 20):
    """This function will put potential "blocks" to a grid array, changing the value of grid.

    Args:
        grid (NDarray): Grid where we want to put blocks in
        val (float): The value of the potential (rho)
        shift (int, optional): The shift of the position of the blocks. Defaults to 20.
    """
    side_length = grid.shape[0]
    mini_side = int(np.sqrt(0.05)*side_length)
    grid[1+shift:mini_side+shift,1+shift:mini_side+shift] = val
    grid[-(mini_side+shift):-(1+shift), -(mini_side+shift):-(1+shift)] = val

# Create a function that solves the Poisson equation
def step(grid, blocks, h):
    """This function is used to evolve the grid based on PDE equation

    Args:
        grid (NDarray): grid that we want to evolve
        blocks (NDarray): The array that contains the potential rho
        h (float): step size, depending on lattice and grid size
    """
    top = grid[0:-2,1:-1]
    bottom = grid[2:,1:-1]
    left = grid[1:-1,0:-2]
    right = grid[1:-1,2:]
    grid[1:-1,1:-1] = (top + bottom + left + right - (h**2)*blocks[1:-1,1:-1])/4 # This is based on PDE equation


# Initial values
N = 101 # Number of data points for each side
a = 20 # The value of the lattice constant
h = a/(N-1)
rho = 1000 # The value of the potential

grid_field = np.zeros((N,N))


block_field = np.zeros((N,N))
place_squares(block_field, val = rho) # Place the blocks

# Setting up the plot
vmax = 0 # The maximum value of the colorbar
vmin = -8000 # The minimum value of the colorbar
fig, ax = plt.subplots(figsize = (10,8))
norm = colors.Normalize(vmin=vmin, vmax= vmax)
im = ax.imshow(grid_field, cmap='cool', norm = norm)
cb = fig.colorbar(im,ax=ax, norm = norm)

print("From the animation, we can see that the system will end in a thermal equilibrium with the walls.\n The potential creates negative value for phi in the area where they are placed")

plt.ion()
plt.show()

# Visualizing using animation
for i in range(20000):
    step(grid_field, block_field, h = h) # Take a step
    if i%20 == 0: # Only draw certain frames to speed up the animation
        fig.canvas.flush_events()
        im.set_data(grid_field)
        fig.canvas.draw()