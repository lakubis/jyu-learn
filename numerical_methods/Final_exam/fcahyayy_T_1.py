# Final exam problem 1
# Author: Felix Cahyadi
# Date: 03.11.2023

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
from scipy.integrate import solve_ivp, solve_bvp, simpson
from scipy.fft import fft, ifft
from numpy.linalg import norm
import fysa1130

# Read data
# Having some trouble reading the data even though it's in the same folder
import os
here = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(here, 'omx.txt')

# We need to skip the header, and set the delimiter to ','
index = np.loadtxt(filename,delimiter=',', skiprows=1).transpose()
date = np.flip(index[0].astype(int))
value = np.flip(index[1])

# Calculate the number of data
N = len(value)
n_array = np.arange(0,N,1)

# Calculate the fourier transform
ft_value = fft(value)

# Set C_k to zero for k>50
ft_value[51:] = 0

# calculate the inverse fourier transform, and taking the absolute value
ifft_value = np.abs(ifft(ft_value))

# Format the date
formatted_date = []
ticks_loc = []
for i in range(7):
    int_date = date[i*200]
    ticks_loc.append(i*200)
    year = int(int_date//1e4)
    month = int((int_date - 10000*year)//100)
    day = int(int_date - 10000*year - 100*month)
    formatted_date.append(f"{year}-{month}-{day}")

# Plot the data
fig, ax = plt.subplots(figsize = (9,5))
ax.plot(n_array, value, label = 'Real data')
ax.plot(n_array, ifft_value, label = 'High frequencies removed')
ax.grid(True)
ax.set_xlabel("Date", fontsize = 16)
ax.set_ylabel("Value", fontsize = 16)
ax.set_xlim([n_array[0], n_array[-1]])
ax.set_xticks(ticks_loc, formatted_date)
#ax.set_ylim([])
ax.legend()

plt.show()

"""
As we can see from the graph, we are removing the higher frequency term from the fft result. After we did some inverse transform to the modified fft, we acquired that the movement of the graph is not as volatile as before. We smoothen the data so that it is more suitable for analysis.
"""