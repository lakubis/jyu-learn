# Last changes 21:41

# Import libraries
import numpy as np


# Define the density function first


def rho(x,y):
    """Generates the value of rho at coordinate (x,y)

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        float: the density at (x,y)
    """
    return (2*y**2 + 1)*(np.sin((4*np.pi*x)/3)+2)

# Part 1
def int_rect(rho,xlim,ylim,part = 50):
    """ Calculate the mass of a rectangle with variable density rho(x,y)

    Args:
        rho (function): density function of the rectangle
        xlim (tuple): tuple containing (xa,xb)
        ylim (tuple): tuple containing (ya,yb)
        part (int, optional): number of partition. Defaults to 50.

    Returns:
        float: the mass of the rectangle
    """
    # extract the value xa,xb,ya,yb
    xa,xb = xlim
    ya,yb = ylim

    # determine hx and hy, the steps in x and y direction
    hx = (xb - xa)/part
    hy = (yb - ya)/part

    # initiate mass
    mass = 0.0

    # Loops, integrating the area
    for i in range(part):
        for j in range(part):
            coor_x = xa+(1/2)*hx+i*hx # generalized midpoint method
            coor_y = ya+(1/2)*hy+j*hy # for both x and y
            mass += rho(coor_x,coor_y)*hx*hy # density times area

    return mass

# output: The mass of the plate is:  43.99919999999981, close to 44 kg
print("The mass of the plate is: ", int_rect(rho,(0,3),(0,2),part = 100))



# Part 2
# We will modify the previous code
def int_triangles(rho,xlim,ylim,part = 50):
    """calculate the mass of the lower and upper triangle in a rectangle with variable density, where there's a line bisecting the triangle from (xa,ya) to (xb,yb)

    Args:
        rho (function): density function of the rectangle
        xlim (tuple): tuple containing (xa,xb)
        ylim (tuple): tuple containing (ya,yb)
        part (int, optional): number of partition. Defaults to 50.

    Returns:
        float: the mass of the lower triangle [(xa,ya),(xb,ya),(xb,yb)]
        float: the mass of the upper triangle [(xa,ya),(xa,yb),(xb,yb)]
    """
    # extract the value xa,xb,ya,yb
    xa,xb = xlim
    ya,yb = ylim

    # determine hx and hy, the steps in x and y direction
    hx = (xb - xa)/part
    hy = (yb - ya)/part

    # calculate the parameters for the line y = mx + c going through (xb,yb) and (xa,ya)
    m = (yb - ya)/(xb - xa)
    c = (ya*xb - xa*yb)/(xb - xa)

    # initiate mass
    mass1 = 0.0
    mass2 = 0.0

    # Loops, integrating the area
    for i in range(part):
        for j in range(part):
            coor_x = xa+(1/2)*hx+i*hx # generalized midpoint method
            coor_y = ya+(1/2)*hy+j*hy # for both x and y

            # decide whether to add the mass to the lower or upper triangle
            if coor_y < (m*coor_x +c): # lower triangle
                mass1 += rho(coor_x,coor_y)*hx*hy # density times area
            elif coor_y > (m*coor_x +c):
                mass2 += rho(coor_x,coor_y)*hx*hy

    return mass1, mass2

mass1, mass2 = int_triangles(rho,(0,3),(0,2),part = 200)

# output: lower triangle mass: 12.20059257386709, upper triangle mass: 31.706014962562882
# output:total mass = 43.90660753642997
print("lower triangle mass: {}, upper triangle mass: {}\ntotal mass = {}".format(mass1,mass2, mass1+mass2))

