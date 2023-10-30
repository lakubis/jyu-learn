# Exercise 7.3
# Author: Felix Cahyadi
# Date: 30.10.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate y_n
def yn(n_arr):
    """ This is the function that calculates the value of y_n

    Args:
        n_arr (NDarray): array containing values of n

    Returns:
        y: array containing the values of y
    """
    N = len(n_arr)
    y = np.sin((np.pi*n_arr)/N)*np.sin((20*np.pi*n_arr)/N)

    return y

# Function to calculate Fourier transform
def fourier_transform(y, n_arr, k_arr):
    """ This is a function that calculates the Fourier transform of a function

    Args:
        y (NDarray): array containing the values of y
        n_arr (NDarray): array containing the values of n
        k_arr (NDarray): array containing the values of k

    Returns:
        ck: The coefficients from Fourier transform
    """

    N = len(n_arr)
    
    ck = 0j*np.zeros_like(k_arr, dtype=np.complex64)
    for i in range(len(ck)):
        for j in range(len(n_arr)):
            ck[i] += y[j]*np.exp(-1j*(2*np.pi*k_arr[i]*n_arr[j])/N)

    return ck


# initiate n and k array
n_arr = np.arange(0,1024,1, dtype=np.complex64)
k_arr = np.arange(0,1024,1, dtype=np.complex64)

# Get y
y = yn(n_arr)

# Do the Fourier transform to get the coefficients
c_ft = fourier_transform(y, n_arr, k_arr)

# Visualize the Fourier transform
fig, ax = plt.subplots(figsize = (9,5))
ax.plot(np.abs(k_arr), np.abs(c_ft))
ax.set_xlim((int(k_arr[0]),int(k_arr[-1])))
ax.grid(True)
ax.set_xlabel("k", fontsize = 16)
ax.set_ylabel("$|c_k|$", fontsize = 16)

plt.show()