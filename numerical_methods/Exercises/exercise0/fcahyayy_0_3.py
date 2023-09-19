import numpy as np
import math

# Create the short array
short_arr = [100, 1e-13, 100, 1e-13, 100, 1e-13, -300, 100, 1e-13, 100, 1e-13,-200,-5e-13]

# Create the long array, 1000 times longer than the short_arr
long_arr = 1000*short_arr

'''
Analytically, the sum of short_arr should be 0, therefore the sum of long_arr should also be 0
'''

# Create kahan_sum()
def kahan_sum(arr):
    """ Function to do Kahan summation, in order to reduce summation error

    Args:
        arr (list): The list of numbers that we want to sum

    Returns:
        float: The compensated sum
    """
    sum = 0.0 # To store the summing result
    c = 0.0 # To store the compensation
    for i in range(len(arr)):
        y = arr[i] - c
        t = sum + y
        c = (t-sum) - y
        sum = t

    return sum

# self-defined Kahan sum
print(kahan_sum(long_arr)) # Output: -1.6830939683131864e-14

# Python sum
print(sum(long_arr)) # Output: 5.685500963055529e-11

# Numpy sum
print(np.sum(long_arr)) # Output: 3.666400516522117e-11

# math sum
print(math.fsum(long_arr)) # Output: 2.5243548967072378e-26, this is the closest one to 0

'''
Out of all the summing function, the math.fsum() function gives the least error. The self-defined Kahan sum gives the second lowest error. It is because of the compensation during the summation
'''

