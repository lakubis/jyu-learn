import numpy as np
from numpy.linalg import norm

def str2arr(coor):
    """This is a function that receives the string coordinate. In this case, we will have "(ax, ay, az, button_state)"

    Args:
        coor (string): Coordinate in string form

    Returns:
        vals_float: The coordinate but in a list of float
    """
    without_brackets = coor[1:-1] # We are removing the brackets, "(ax, ay, az, button_state)" -> "ax, ay, az, button_state"
    vals = without_brackets.split(",") # Split the string from "ax, ay, az, button_state" -> ["ax", "ay", "az", "button_state"]

    # Sometimes it does not able to convert, I don't know why, we're going to throw a try-except case here
    try:
        vals_float = [float(val) for val in vals] # Here, we are going to transform ["ax", "ay", "az", "button_state"] -> [ax, ay, az, button_state]
    except:
        vals_float = [0.0,0.0,0.0,0.0] # To prevent errors, we return zeros instead.
    return vals_float

def calibrate(arduino, sample_size = 50):
    
    # Counter to control the while loop
    step = 0

    a_list = []

    while step<sample_size:
        message = str(arduino.readline())
        
        if message!= "b''":
            preprocess = message.split("'")[1:-1][0].split(";")[:-1]
            states = [str2arr(coor) for coor in preprocess]

            for state in states:
                if len(state) == 4:
                    now_state = state
                    step+=1

                accel_mg = np.array(now_state[:-1])
                accel = 9.81e-3*accel_mg
                
                a_list.append(accel)

    a_numpy = np.array(a_list)
    a_mean = np.sum(a_numpy, axis=0)/a_numpy.shape[0]
    offset = a_mean - np.array([0,0,9.81]) # Check this part

    key_list = ['num_data', 'accelerations', 'offset']
    values_list = [step, a_list, offset]
    calibration_info = dict(zip(key_list,values_list))

    return calibration_info



