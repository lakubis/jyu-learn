#!/usr/bin/env python3

"""
Running a simulation and attaching measurements to it.
A measurement has a frequency that controls how often it's performed,
and a method _measure(self,simulation,*args,**kwargs). 
The "simulation" is just random number generation.   
"""

import numpy as np

class MeasurementBase:
    def __init__(self):
        # clear list of measuments and set number of collected data to zero
        self.measurements = [] 
        self.steps = 0
        print('MeasurementBase __init__')

    def add_measurement(self, measurement, frequency = 1, *args, **kwargs):
        # adds a callback function "measurement._measure"
        # default: frequency = 1 means measure at every step

        # test the function has __call__ method
        if not callable(measurement._measure):
            print(f"{measurement.name} has no callable method measure")
            raise ValueError

        print(f'adding new measurement: name={measurement.name} frequency={frequency}')
        self.measurements.append((measurement._measure, frequency, args, kwargs))

    def do_measurements(self):
        self.steps +=1
        for measure, frequency, args, kwargs in self.measurements:
            if self.steps%frequency == 0:
                measure(self,*args,**kwargs)
        


class Measure_Average(MeasurementBase):
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.average=0.0
        self.N=0
        self.name='average'        
        if(self.verbose): print('Measure_Average init: zeroing average')

    def _measure(self,simulation,*args,**kwargs):
        if(self.verbose): print(f'  {self.name:10s} using {simulation.data}') 
        self.average += simulation.data
        self.N +=1
        
    def get(self):
        return self.average/self.N

class Measure_Limits(MeasurementBase):
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.lower = 1e100
        self.upper = -1e100
        self.name='limits'        

    def _measure(self,caller,*args,**kwargs):
        if(self.verbose): print(f'  {self.name:10s} using data {caller.data}') 
        self.lower = min(self.lower,caller.data)
        self.upper = max(self.upper,caller.data)
        
    def get(self):
        return self.lower,self.upper
        
class Simulation(MeasurementBase):
    def __init__(self, shift=0, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.shift = shift
        self.data  = 0.0
        print('Simulation init: setting shift = ',shift)
        

    def run(self,N):
        for _ in range(N):
            # the next line *is* the simulation :)
            self.data = np.random.random()+self.shift            
            self.do_measurements()

         

    

if __name__ == '__main__':
    
    average = Measure_Average() # you may add verbose=True
    limits  = Measure_Limits()  # you may add verbose=True
    simu = Simulation(shift=3.0)
    simu.add_measurement(average)
    simu.add_measurement(limits,frequency=10)

    for i in range(5):
        print(u'\u2500'*80)  # horizontal line
        print('iter ',i)
        simu.run(500)
        ave = average.get()
        l,u = limits.get()
        print(f' average = {ave}')
        print(f' lower, upper = {l,u}')
