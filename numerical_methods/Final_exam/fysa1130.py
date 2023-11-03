import numpy as np
from scipy.linalg import solve


def numerical_derivative(points, h, func, order=1):
    '''
    Compute numerical derivative of a given function

    Parameters
    ----------
    points : numpy array
        points at which the function is evaluated
    h : float
        h in finite difference equation
    func : function
        function which is differentiated
    order : int, optional
        order of accuracy of FD method. The default is 1.

    Returns
    -------
    Numerical derivative of a given function at given points.
    '''

    if order == 1:
        return (func(points + h) - func(points)) / h
    elif order == 2:
        return (func(points + h/2) - func(points - h/2)) / h
    else:
        raise ValueError('Wrong order of accuracy.')


def rms(data1, data2):
    '''
    Calculates the root mean square of the difference of two given datasets

    Parameters
    ----------
    data1 : numpy array

    data2 : numpy array


    Returns
    -------
    root mean square

    '''
    return np.sqrt(np.mean((data1-data2)**2))


def numerical_derivative_var_h(points, data):
    '''
    Computes the numerical derivative for given data points.

    Parameters
    ----------
    points : list
        x-value of the data points
    data : list
        data values

    Returns
    -------
    numerical derivative of the given data

    '''
    der = np.zeros_like(points)
    hp = np.zeros_like(points)
    hm = np.zeros_like(points)

    hm[1:] = points[1:] - points[0:-1]
    hp[0:-1] = points[1:] - points[0:-1]

    # points except first and last
    der[1:-1] = hm[1:-1]/(hp[1:-1] + hm[1:-1]) * \
        (data[2:] - data[1:-1])/hp[1:-1] + \
        hp[1:-1]/(hp[1:-1] + hm[1:-1]) * \
        (data[1:-1] - data[0:-2])/hm[1:-1]

    # first and last points
    der[0] = (data[1] - data[0])/hp[0]
    der[-1] = (data[-1] - data[-2])/hm[-1]

    return der


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


def trapezoidal_adaptive(func, imin, imax, tol=1e-12, max_iter=10):
    '''
    Integrates given function by using adaptive trapezoidal

    Parameters
    ----------
    func : function
        function which will be integrated
    imin : float
        lower integration limit
    imax : float
        upper integration limit
    tol : float
        tolerance limit
    max_iter : int, optional
        maximum number of iterations. The default is 10.

    Returns
    -------
    myval : float
        value of the integral
    '''

    h = imax - imin
    myval = (h/2)*(func(imin) + func(imax))

    for k in range(0, max_iter):
        h = h/2
        n = 2**k
        new_vals = np.sum(func(np.linspace(imin+h, imax-h, n)))
        prev = myval
        myval = prev/2 + h*new_vals

        if abs(myval - prev) < tol:
            return myval

    print('Didn\'t converge')
    return (myval, abs(myval - prev))


def semiopen_trapezoidal(func, imin, imax, N):
    '''
    Integrate by using trapezoidal rule with open end on the left

    Parameters
    ----------
    func : function
        function which will be integrated
    imin : float
        lower integration limit, not necessary defined
    imax : float
        upper integration limit
    N : int
        number of points where function is evaluated

    Returns
    -------
    myval : float
        value of the integral
    '''

    X, h = np.linspace(imin, imax, N, retstep=True)
    weight = 12*np.ones(N-1)
    weight[-1] = 6
    weight[0:3] = [29, -4, 17]
    myvals = func(X[1:])
    myval = (h/12) * np.dot(myvals, weight)

    return myval


def jacobian_matrix(func, x, h=1e-6):
    '''
    Parameters
    ----------
    func : function

    x : float
        point where Jacobian is formed
    h : float, optional
        step size. The default is 1e-6.

    Returns
    -------
    J : numpy array
        Jacobian matrix
    '''

    n = len(x)
    J = np.zeros((n, n))

    for i in range(0, n):
        for j in range(0, n):
            H = np.zeros(n)
            H[j] = h
            F_plus = func(x + H)
            F_minus = func(x - H)
            J[i, j] = (F_plus[i] - F_minus[i])/(2*h)

    return J


def newton_raphson(func, x0, tol=1e-6, max_iter=20):
    '''
    N-dimensional Newton-Raphson solver

    Parameters
    ----------
    func : function
        system of nonlinear equations
    x0 : numpy array
        initial guess (n-dimensions)
    tol : float, optional
        Tolerance. The default is 1e-6.
    max_iter : int, optional
        Maximum numer of iterations. The default is 20.

    Returns
    -------
    Solution
    '''
    x = x0

    for i in range(0, max_iter):
        myvals = func(x)
        J = jacobian_matrix(func, x)
        deltax = solve(J, -myvals)
        x = x + deltax
        if np.linalg.norm(func(x)) < tol:
            return x
    print('Didn\'t converge')
    return x


def euler_method(F, a, b, y0, h):
    '''
    Solving for n:th order DE by using Euler's method

    Parameters
    ----------
    F : function
        Derivatives of the DE
    a : float
        lower limit
    b : float
        upper limit
    y0 : numpy array
        initial values
    h : float
        step size

    Returns
    -------
    x_step : numpy array
        evaluation points
    y_step : numpy array
        solution and its derivatives at given x
    '''

    x = a
    y = y0
    x_step = [x]
    y_step = [y]

    while x < b:
        y = y + F(x, y)*h
        x = x + h

        x_step.append(x)
        y_step.append(y)

    return np.asarray(x_step), np.asarray(y_step)


def rk4(dydx, x0, x1, y, h):
    '''
    Fourth order Runge-Kutta ODE integrator

    Parameters
    ----------
    dydx : function
        Specifies the 1st-order differential equations
        F(x,y) = [dy1/dx dy2/dx dy3/dx ...].
    x0 : float
        lower limit for x
    x1 : TYPE
        upper limit for x
    y : array like
        initial values
    h : float
        step size

    Returns
    -------
    xsol : numpy array
        Values of x at which solution is computed.
    ysol : numpy array
        Values of y corresponding to the x-values.

    '''
    # Initialize
    x = float(x0)
    y = y.astype('float')

    # List to gather output data
    xsol = [x]
    ysol = [y]

    done = False
    while not done:
        if x+h > x1:
            h = x1-x
            done = True
        K1 = h*dydx(x, y)
        K2 = h*dydx(x+0.5*h, y+0.5*K1)
        K3 = h*dydx(x+0.5*h, y+0.5*K2)
        K4 = h*dydx(x+h, y+K3)

        x = x + h
        y = y + (K1 + 2*K2 + 2*K3 + K4)/6

        xsol.append(x)
        ysol.append(y)

    # Format output as arrays
    xsol = np.array(xsol)
    ysol = np.array(ysol)

    return xsol, ysol


def fdjac(func, x, step):
    '''Finite difference Jacobian

    Arguments:
      func:  Function, which returns F_i values at x (single input argument),
             where x is a numpy array of values x_1,x_2,...,x_n
      x:     Point where Jacobian is evaluated
      step:  Step size for finite difference calculation

    Returns:
      J:     Jacobian
    '''

    n = len(x)
    J = np.empty((n, n))
    for i in range(n):
        b = np.zeros(n)
        b[i] = step
        J[:, i] = (func(x+b) - func(x-b)) / (2*step)

    return J


def gnewton(func, x0, tol=1e-6, step=1e-8, geps=1e-50, jac=None, diag=False, full_output=False, Niter=50):
    '''Globally convergent Newton-Raphson solver for finding roots
    of simultaneous equations in form F_i(x_1,x_2,...,x_n) = 0, where
    i = 1,2,...,n

    Arguments:
      func:  Function, which returns F_i values at x (single input argument),
             where x is a numpy array of values x_1,x_2,...,x_n
      x0:    Starting values for iteration
      tol:   Tolerance
      jac:   Jacobian matrix containing derivatives J=dF_i/dx_j at x (single 
             input argument)
      diag:  Diagnostics printed on standard output
      full_output: If iteration points are returned

    Returns:
      x:     Estimation of root location
      xi:    Locations of iteration points (returned if full_output=True)
    '''

    n = len(x0)
    x = x0.astype('float')
    xi = []

    F = func(x)
    fnorm = np.linalg.norm(F)

    if diag:
        print('Globally convergent Newton-Raphson:')
        print('{:>20s} {:>20s} {:>20s}'.format('fnorm', 'dxnorm', 't'))
        print('{:>20e}'.format(fnorm))
    if full_output:
        xi.append(x)

    for iter in range(Niter):

        if jac == None:
            J = fdjac(func, x, step)
        else:
            J = jac(x)
        dx = np.linalg.solve(J, -F)
        dxnorm = np.linalg.norm(dx)

        if dxnorm < tol:
            if diag:
                print('{:>20e} {:>20e}'.format(fnorm, dxnorm))
                print('Done')
            if full_output:
                return x, np.array(xi)
            return x

        xold = x
        fnorm_old = fnorm
        t = 1.0
        while t >= geps:
            x = xold + t*dx
            F = func(x)
            fnorm = np.linalg.norm(F)
            if fnorm < fnorm_old:
                break
            t *= 0.5
        if full_output:
            xi.append(x)

        if diag:
            print('{:>20e} {:>20e} {:>20e}'.format(fnorm, dxnorm, t))

        if t < geps:
            raise ValueError('Could not find step size which minimizes residual')

        if fnorm < tol:
            if diag:
                print('Done')
            if full_output:
                return x, np.array(xi)
            return x

    raise ValueError('Globally convergent Newton-Raphson did not converge')


def golden_search(func, a, b, tol=1e-6):
    '''
    Minimization using golden search

    Parameters
    ----------
    func : function
        The function which is minimized
    a : float
        Lower limit
    b : float
        Upper limit
    tol : float, optional
        Tolerance. The default is 1e-6.

    Returns
    -------

    x : float
        Minimum point
    f : float
        Function value at minimum
    '''

    # Initialize
    niter = int(np.ceil(-2.078087*np.log(tol/abs(b-a))))
    R = 0.618033989
    C = 1.0 - R

    # Inner points
    x1 = R*a + C*b
    x2 = C*a + R*b
    f1 = func(x1)
    f2 = func(x2)

    for i in range(niter):
        if f1 > f2:
            a = x1
            x1 = x2
            f1 = f2
            x2 = C*a + R*b
            f2 = func(x2)
        else:
            b = x2
            x2 = x1
            f2 = f1
            x1 = R*a + C*b
            f1 = func(x1)

    if f1 < f2:
        return x1, f1
    return x2, f2


def gold_bracket(func, x1, h):
    '''
    Search for limits for golden search

    Parameters
    ----------
    func : function
        Function which is minimized
    x1 : float
        Initial guess
    h : float
        Step size.

    Raises
    ------
    ValueError
        If no minimum is found

    Returns
    -------
    float
        x smaller than minimum
    flaot
        x larger than minimum
    '''
    R = 1.618033989
    f1 = func(x1)
    x2 = x1 + h
    f2 = func(x2)

    if f2 > f1:
        h = -h
        x2 = x1 + h
        f2 = func(x2)

        # If minimum is between x1-h and x1+h
        if f2 > f1:
            return x2, x1-h

    # Search loop
    for i in range(100):
        h = R*h
        x3 = x2 + h
        f3 = func(x3)
        if f3 > f2:
            return x1, x3
        x1 = x2
        f1 = f2
        x2 = x3
        f2 = f3

    raise ValueError('Did not find a minimum')


def powell(func, x, h=1, tol=1e-6):
    '''Powell algorithm for minimization

    Arguments:
      func:    Function, which returns a scalar value with a vector
               argument x.
      x:       Starting values for iteration
      h:       Initial step size for line search
      tol:     Tolerance

    Returns:
      x:       Vektor of paramters, which minimize the function func
    '''

    n = len(x)          # Number of dimensions
    df = np.zeros(n)    # Change of function value
    u = np.identity(n)  # Direction vectors

    for j in range(100):
        xold = x.copy()
        fold = func(xold)

        # Line search along the vector
        for i in range(n):
            v = u[i]

            def f(s):
                return func(x + s*v)

            a, b = gold_bracket(f, 0.0, h)
            s, fmin = golden_search(f, a, b)
            df[i] = fold - fmin
            fold = fmin
            x = x + s*v

        # Last line search
        v = x-xold

        def f(s): return func(x + s*v)

        a, b = gold_bracket(f, 0.0, h)
        s, flast = golden_search(f, a, b)
        x = x + s*v

        if np.sqrt(np.dot(x - xold, x-xold)/n) < tol:
            return x

        # The largest change and new directions
        imax = np.argmax(df)
        for i in range(imax, n-1):
            u[i] = u[i+1]
        u[n-1] = v

    raise ValueError('Powell did not converge')