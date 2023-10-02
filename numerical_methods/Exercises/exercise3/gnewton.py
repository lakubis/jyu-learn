import numpy as np


# Finite difference Jacobian
def fdjac(func,x,step):
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
    J = np.empty((n,n))
    for i in range(n):
        b = np.zeros(n)
        b[i] = step
        J[:,i] = (func(x+b) - func(x-b)) / (2*step)

    return J


def gnewton(func,x0,tol=1e-6,step=1e-8,geps=1e-50,jac=None,diag=False,full_output=False,Niter=50):
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
        print('{:>20s} {:>20s} {:>20s}'.format('fnorm','dxnorm','t'))
        print('{:>20e}'.format(fnorm))
    if full_output:
        xi.append(x)

    for iter in range(Niter):

        if jac == None:
            J = fdjac(func,x,step)
        else:
            J = jac(x)
        dx = np.linalg.solve(J,-F)
        dxnorm = np.linalg.norm(dx)

        if dxnorm < tol:
            if diag:
                print('{:>20e} {:>20e}'.format(fnorm,dxnorm))
                print('Done')
            if full_output:
                return x,np.array(xi)
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
            print('{:>20e} {:>20e} {:>20e}'.format(fnorm,dxnorm,t))
            
        if t < geps:
            raise ValueError('Could not find step size which minimizes residual')

        if fnorm < tol:
            if diag:
                print('Done')
            if full_output:
                return x,np.array(xi)
            return x

    raise ValueError('Globally convergent Newton-Raphson did not converge')
