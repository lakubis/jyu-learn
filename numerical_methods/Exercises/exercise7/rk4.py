import numpy as np

def rk4( dydx, x0, x1, y, h ):
    '''Fourth order Runge-Kutta ODE integrator
    
    Arguments:
      dydx:  Function that specifies the 1st-order differential equations
             F(x,y) = [dy1/dx dy2/dx dy3/dx ...].
      x0:    Initial value of x.
      x1:    Final value of x.
      y:     Initial y values.
      h:     Step size for x (last step size might be smaller to hit x1 exactly).
    Returns:
      xsol:  Values of x at which solution is computed.
      ysol:  Values of y corresponding to the x-values.
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
        K1 = h*dydx(x,y)
        K2 = h*dydx(x+0.5*h,y+0.5*K1)
        K3 = h*dydx(x+0.5*h,y+0.5*K2)
        K4 = h*dydx(x+h,y+K3)

        x = x + h
        y = y + (K1 + 2*K2 + 2*K3 + K4)/6

        xsol.append(x)
        ysol.append(y)

    # Format output as arrays
    xsol = np.array(xsol)
    ysol = np.array(ysol)
    
    return xsol,ysol