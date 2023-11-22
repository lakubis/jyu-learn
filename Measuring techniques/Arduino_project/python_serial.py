# Importing Libraries 
import serial 
import time 
arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1) 

def write_read(x): 
	arduino.write(bytes(x, 'utf-8')) 
	time.sleep(0.05) 
	data = arduino.readline() 
	return data 

def just_read():
	data = arduino.readline()
	return data

while True: 
	#num = input("Enter a number: ") # Taking input from user 
	value = just_read() 
	print(value) # printing the value