import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

time_array = np.linspace(-pi, pi, 50)
sin_array = np.sin(time_array)

plt.plot(time_array, sin_array)
plt.show()