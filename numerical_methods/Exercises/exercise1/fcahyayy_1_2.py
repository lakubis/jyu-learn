# Here, we want to calculate the second derivative of a function given a list of array y

# Last changes: 21:41

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# define x array and dx
xarr, dx = np.linspace(-np.pi,np.pi,1000,retstep= True)

# define y array
yarr = np.sin(xarr)

def second_derivative(yarr,dx):
    """ This is a function that gives the second derivative of a uniformly distributed array y

    Args:
        yarr (np.array): uniformly distributed array y
        dx (float): the x difference between data points

    Returns:
        np.array: an array containing the second derivative of the y array
    """
    
    # create an array to store the second derivative of y
    ddyarr = np.zeros_like(yarr)

    # Treat the center first
    for i in range(1,len(ddyarr)-1):
        ddyarr[i] = (yarr[i-1] - 2*yarr[i] + yarr[i+1])/(dx**2)

    # treat the first points using forward difference
    ddyarr[0] = (1/dx**2)*(yarr[2] - 2*yarr[1] + yarr[0])

    # treat the last point using backward difference
    ddyarr[-1] = (1/dx**2)*(yarr[-1] - 2*yarr[-2] + yarr[-3])
    
    return ddyarr

# get the value
ddyarr = second_derivative(yarr,dx)
print("This is the result for the second derivative: ", ddyarr)

# quantify the error
error = (ddyarr.copy()-(-np.sin(xarr)))
print("This is the error array: ",error)

# from here, we can see that the errors of the center points are about 1e-8, while the error of the edge cases are about 1e-3. The accuracy of the derivative drops on the edge cases

print("from here, we can see that the errors of the center points are about 1e-8, while the error of the edge cases are about 1e-3. The accuracy of the derivative drops on the edge cases")

# plot the values
fig, ax = plt.subplots(figsize = (8,5))
ax.plot(xarr, -np.sin(xarr), label = "numpy -sin(x)")
ax.plot(xarr, ddyarr, label = "finite difference method")
ax.legend()
ax.set_xlim((-np.pi,np.pi))
ax.grid(True)
plt.show()


    