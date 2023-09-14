# First, we are going to test the math library
import math
x = 1e-5

# Calculate math.exp(x) - 1
print(math.exp(x) - 1) # Output: 1.0000050000069649e-05

# Calculate math.expm1(x)
print(math.expm1(x)) # Output: 1.0000050000166667e-05

'''
The value that we get using math.exp(x)-1 is different from math.expm1(x) for small value of x. It is probably because we substract two numbers of similar magnitude, and hence created a huge error in the substraction.
'''

# Create a function to calculate exp(x) - 1 using the taylor series
def expMinusOne(x,n):
    """ A function to calculate exp(x) - 1 using Taylor series of certain order n

    Args:
        x (float): The value of x that we want to calculate
        n (int): The order of the Taylor series

    Raises:
        Exception: n should be integer and should be more or equal to 1

    Returns:
        float: The result of the Taylor series
    """


    result = 0.0 # To store the result of the computation
    if n < 1 or type(n)!=int:
        raise Exception('n should be integer and >=1')
    
    for i in range(1,n+1):
        result += (x**i)/math.factorial(i)

    return result

# Calculate expMinusOne
print(expMinusOne(x,3)) # Output: 1.0000050000166668e-05

'''
The result from the Taylor series is close to the value from the math library with just the third order.
'''


