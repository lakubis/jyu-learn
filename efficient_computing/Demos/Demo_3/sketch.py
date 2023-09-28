"""
 Container of boiling hot water is dipped to liquid nitrogen (-196 C)
 top of the container is held at room temperature (22 C)
 Crude simulation of heat transfer in 2D using height x width finite size elements
"""
import numpy as np

class Heat2D:
    def __init__(self, height, width):
        # index order in heat_map is [y,x] to make matshow show it correctly
        self.heat_map = np.zeros((height+2,width+2),dtype=np.float64)
        self.heat_map[:] = 100.0 # initial temperature
        # dip to very cold environment, top stays at room temperature
        self.heat_map[:,0]  = -196.00
        self.heat_map[:,-1] = -196.00
        self.heat_map[-1,:] = -196.00
        self.heat_map[0,:]  = 22.0  
    
    def step(self):        
        mid    = self.heat_map[1:-1,1:-1]
        above  = self.heat_map[:-2,1:-1]
        below  = self.heat_map[2:,1:-1]
        right  = self.heat_map[1:-1,:-2]
        left   = self.heat_map[1:-1,2:]
        mid[:] = (mid+above+below+left+right)/5
        return mid


def update(data):
    mat.set_data(data)

def data_gen():
    yield heat.step()
        
if __name__=='__main__':
    
    # initialize 100x100 element water container        
    heat = Heat2D(100,100)
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots()
    mat = ax.matshow(heat.heat_map,cmap='cool',vmin=-200.0, vmax=100.0)
    plt.colorbar(mat)
    ani = animation.FuncAnimation(fig, update, data_gen, interval=10, save_count=500)
    plt.show()
