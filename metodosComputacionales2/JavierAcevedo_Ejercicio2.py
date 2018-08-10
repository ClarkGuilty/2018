#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 06:53:04 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Create an bidimensional array of random numbers with shape (4,8).
#First, set the last column to -1 and then set the second row to 2.
print("Ex 1.1")
array = np.zeros((4,8))
array[1,:] = 2
array[:,-1] = -1
print(array)
print('\n')

#Using a=np.random.normal(size=1000) generate an array of 1000 thousand random numbers generated from a normal (i.e. gaussian) distribution with mean zero and standard deviation of one.
#Print the number of elements with values larger than 2.0. Is this number close to what you expected from the properties of a gaussian distribution?
print("Ex 1.2")
array = np.random.normal(size = 1000)
index = (array >2.0)
print(array[index])
print("El número de elementos mayor a 2 es coherente con la naturaleza gaussiana de la distribución")
print('\n')


#Using a=np.random.normal(size=1000) generate an array of 1000 thousand random numbers generated from a normal (i.e. gaussian) distribution with mean zero and standard deviation of one.
#Then using only ufuncs on a generate a new array b that is -1 wherever a is negative and 1 wherever a is positive.
print("Ex 1.3")
array = np.random.normal(size = 1000)
b = array / np.abs(array)
print(b)
print('\n')


#Make a plot of the Lissajous curve corresponding to a=5, b=4.
print("Ex 4.1")
x = np.linspace(0,2*np.pi, 1000)
xL = np.sin(5*x)
yL = np.sin(4*x)
plt.figure()
plt.plot(xL,yL)
plt.title("Curva de Lissajous con a = 5 y b = 4")
print('\n')


#Write a function to generate N points homogeneously distributed over a circle of radius 1 centered on the origin of the coordinate system. The function must take as a in input N and return two arrays x and y with the cartesian coodinates of the points. Use the function and generate 1000 points to make a plot and confirm that indeed the points are homogeneously distributed.
#Write a similar function to distribute N points over the surface of the 3D sphere of radius 1.

print("Ex 4.2")
def fun2d(N):
    x = np.linspace(0,2*np.pi, N)
    A = np.random.uniform(size =N)
    xC = A*np.sin(x)
    yC = A*np.cos(x)
    return xC, yC

x,y = fun2d(1000)
plt.figure()
plt.scatter(x,y)

def fun3d(N):
    #theta = np.linspace(0,2*np.pi, N)
    #phi = np.linspace(0,np.pi, N)
    theta = np.random.uniform(size = N)*2*np.pi
    phi = np.random.uniform(size = N)*np.pi
    A = 1
    xC = A*np.sin(theta)*np.sin(phi)
    yC = A*np.cos(theta)*np.sin(phi)
    zC = A*np.cos(phi)
    return xC, yC, zC


x,y,z = fun3d(1000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c='r', marker = 'o')
















