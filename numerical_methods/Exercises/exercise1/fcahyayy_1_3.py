# Integration using Simpsons 3/8 rule.

# last changes: 22:02

# Import libraries
import numpy as np
from scipy.integrate import quad

def simps38(f, xlim, N):
    """ integration using Simpson's 3/8 rule

    Args:
        f (function): function that we want to integrate
        xlim (tuple): tuple containing the limit of the integration (xa,xb)
        N (int): number of nodes, must be in form of 4+i*3, where i is a positive integer

    Raises:
        Exception: If N does not comply with the condition

    Returns:
        float: the result of the integration 
    """

    iter = 0
    # Check if the number of node is right and how many times the iteration should be done
    if N == 4:
        iter = 1 # 1-time process
    elif (N-4)%3 == 0:
        iter = int((N-4)/3) + 1 # extra iteration, ex. N = 10 means 3 iterations
    else:
        raise Exception('N should be 4 or 4+i*3, where i is a positive integer')
    
    # create an array for x, and get the value of h
    xarr, h = np.linspace(xlim[0],xlim[1], N, retstep=True)

    # initialize summation
    integral = 0.0

    # iteration for calculating the integral
    for i in range(iter):
        j = i*3 # multiply the index by 3 because we are using Simpson's 3/8 rule
        integral += (h/8)*(3*f(xarr[j]) + 9*f(xarr[j+1]) + 9*f(xarr[j+2]) + 3*f(xarr[j+3])) # Simpson's 3/8 rule

    return integral


# Integrate sin x from 0 to pi
f = lambda x: np.sin(x)
xlim = (0,np.pi)
result_simps = simps38(f,xlim,13)
result_scipy = quad(f,xlim[0],xlim[1])

print("Result from Simpson 3/8: ", result_simps)
print("Result from SciPy: ", result_scipy[0])

# Finding the number of nodes N where the relative error is less than 1e-4
def find_node_simpson(f, f_simp, xlim, tol = 1e-4, max_iter = 100):
    """ Finding the number of nodes needed to reduce the error below the tolerance

    Args:
        f (function): the function that we want to integrate
        f_simp (function): the simpson function that we made
        xlim (tuple): the limits of integration (xa, xb)
        tol (float, optional): the error tolerance that we want to achieve. Defaults to 1e-4.
        max_iter (int, optional): the maximum number of iteration. Defaults to 100.

    Returns:
        float: the minimum number of node needed to reduce the error below the tolerance
    """
    
    for i in range(max_iter):
        N = 4 + 3*i # the number of nodes
        eta = np.abs(1-(f_simp(f,xlim,N)/quad(f,xlim[0],xlim[1])[0])) # the relative error
        if eta < tol: # if the relative error is less than the tolerance
            return N, eta
    
# Find the minimum number of node
N_min, eta_min = find_node_simpson(f, simps38, xlim)
print("The minimum number of nodes to get the error under 1e-4 is: ", N_min)
print("The relative error in this case is: ", eta_min)
    
# calculate the relative error between fysa1130.trapezoidal and scipy quad using N = 13

# trapezoidal function from fysa1130
def trapezoidal(func, imin, imax, N):
    '''
    Integrate by using trapezoidal rule

    Parameters
    ----------
    func : function
        function which will be integrated
    imin : float
        lower integration limit
    imax : float
        upper integration limit
    N : int
        number of points where function is evaluated

    Returns
    -------
    myval : float
        value of the integral
    '''
    # weights or coefficients
    weight = 2*np.ones(N)
    weight[0] = 1
    weight[-1] = 1

    X, h = np.linspace(imin, imax, N, retstep=True)
    myvals = np.array(func(X))
    myval = (h/2)*np.dot(weight, myvals)

    return myval

print("Using N = 13 for the trapezoidal function, the relative error is: ", np.abs(1-(trapezoidal(f,xlim[0],xlim[1],N =13)/quad(f,xlim[0],xlim[1])[0])))

print("Using the same number of node, the simpson 3/8 rule gives a much better result")


