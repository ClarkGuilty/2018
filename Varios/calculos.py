#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:37:13 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from mpl_toolkits.mplot3d import Axes3D



densidadTheo = np.loadtxt('density0.dat')
potTheo = np.loadtxt('potential0.dat')
potReal = np.loadtxt('potential1.dat')
fCreal = np.loadtxt("fdens0.dat")
fCimag = np.loadtxt("fdens1.dat")
fCtotal = fCreal**2 + fCimag**2
dtest = densidadTheo.copy()
Nx = 128
xmin = -1.0
xmax = 1.0
#xmin = 0
#xmax = 2.0
L = xmax-xmin
x = np.linspace(xmin,xmax,Nx, endpoint = False)


def pot(inp, ino):
    return -np.cos(0.5*np.pi*inp)*np.cos(0.5*np.pi*ino)

def dens(xin,yin):
    return -np.pi*pot(xin,yin)/8.0
    #return -pot(xin,yin)

def simpleExample(xin,yin, nx,ny):
    return np.sin(np.pi*xin*nx*2.0/L)+np.sin(np.pi*yin*ny*2.0/L)

def simpleExample2(xin,yin, nx,ny):
    return -( (2.0*nx*np.pi/L)**2*np.sin(np.pi*xin*nx*2.0/L)+(2.0*ny*np.pi/L)**2*np.sin(np.pi*yin*ny*2.0/L))/(4*np.pi)


t = np.linspace(xmin,xmax,Nx, endpoint = False)
#bump = np.exp(-0.05*t**2)

#bump = bump / np.trapz(bump) # normalize the integral to 1
test = np.zeros((Nx,Nx))
totalMass = 0
for i in range(Nx):
    for j in range(Nx):
        #test[i][j] = dens(x[i],t[j])
        test[i][j] = simpleExample2(x[i],t[j],3,20)
        dtest[i][j] = test[i][j]
        totalMass += test[i][j]*(2.0/128)**2

death = densidadTheo - test
totalDeath = death.sum().sum()



testOld = test.copy()
test -= np.ones((Nx,Nx))*totalMass/(xmax-xmin)**2 #es importante que la freq 0 sea 0
img_ft = fft.fft2(densidadTheo)
freq = fft.fftfreq(len(test),1.0/128)
fimag = np.imag(img_ft)/Nx**2
freal = np.real(img_ft)/Nx**2
ftotal = freal**2 + fimag**2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(xmin, xmax, 2.0/128)
X, Y = np.meshgrid(x, y)
#zs = np.array([dens(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#zs = np.ravel(np.log(real**2+imag**2))
zs = np.ravel(ftotal)
Z = zs.reshape(X.shape)
plt.title("Python")

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#
#
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(xmin, xmax, 2.0/128)
X, Y = np.meshgrid(x, y)
#zs = np.array([dens(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
#zs = np.ravel(np.log(real**2+imag**2))
zs = np.ravel(fCtotal)
Z = zs.reshape(X.shape)
plt.title("C")
ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#


imgC = fCreal + 1j * fCimag
testBack = np.real(fft.ifft2(img_ft))
testBackC = np.real(fft.ifft2(imgC))*Nx**2
testBack += np.ones((Nx,Nx))*totalMass/(xmax-xmin)**2 #es importante que la freq 0 sea 0

plt.figure()
plt.imshow(testBack)
plt.title("Python")
cbar = plt.colorbar()

plt.figure()
plt.imshow(np.real(testBackC))
plt.title("C")
cbar = plt.colorbar()

#plt.figure()
#plt.imshow(np.real(testBackC*Nx**2-testBack))
#cbar = plt.colorbar()

def fastShow(image, title="none"):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()

totalImg = imgC-img_ft
ddiff = densidadTheo - dtest
fastShow(ddiff, title = "diff")




#h.set_tight_layout(False)
#plt.pcolor(testOld)
#cbar = plt.colorbar()

#plt.imshow(testOld)
#cbar = plt.colorbar()
#plt.title('densidad python')
#plt.figure()
#plt.imshow(testOld)
#cbar = plt.colorbar()
#plt.title('densidad C')

#plt.figure()
#plt.imshow(densidadTheo)
#plt.figure()
#plt.imshow(test)













