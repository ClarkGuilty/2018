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
fpotCreal = np.loadtxt("fpot0.dat")
fpotCimag = np.loadtxt("fpot1.dat")

fCtotal = fCreal**2 + fCimag**2
dtest = densidadTheo.copy()
Nx = 128
xmin = 0
xmax = 2.0
#xmin = 0
#xmax = 2.0
PI = np.pi
G = 1.0
L = xmax-xmin
dx = L/Nx
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
        potTheo[i][j]=simpleExample(x[i],t[j],3,20)

death = densidadTheo - test
totalDeath = death.sum().sum()



testOld = test.copy()
test -= np.ones((Nx,Nx))*totalMass/(xmax-xmin)**2 #es importante que la freq 0 sea 0
img_ft = fft.fft2(test)
freq = fft.fftfreq(len(test),1.0/128)
fimag = np.imag(img_ft)/Nx**2
freal = np.real(img_ft)/Nx**2
ftotal = freal**2 + fimag**2



#
def fast3d(image, title="none"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(xmin, xmax, 2.0/128)
    X, Y = np.meshgrid(x, y)
    #zs = np.array([dens(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    #zs = np.ravel(np.log(real**2+imag**2))
    zs = np.ravel(image)
    Z = zs.reshape(X.shape)
    plt.title(title)
    ax.plot_surface(X, Y, Z)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

#fast3d(ftotal, "antes python")
    



imgC = fCreal + 1j * fCimag


testBack = np.real(fft.ifft2(img_ft))
testBackC = np.real(fft.ifft2(imgC))*Nx**2
testBack += np.ones((Nx,Nx))*totalMass/(xmax-xmin)**2 #es importante que la freq 0 sea 0


def calcK3(i2,j2):
    if ((j2 == 0) and (i2 == 0)):
        return 1
    if(i2<Nx/2+1):
        i2 = PI*i2;
    if(j2<Nx/2+1):
        j2 = PI*j2;
    if(i2>=Nx/2+1):
        i2 = -PI*i2;
    if(j2>=Nx/2+1):
        j2 = -PI*j2;
    rta1 = 2*np.sin(dx*i2)/dx
    rta2 = 2*np.sin(dx*j2)/dx
    return 1.0/(np.power(rta1,2)+np.power(rta2,2));
    
def calcK5(i2,j2):
    if ((j2 == 0) and (i2 == 0)):
        return 1
    if(i2<Nx/2+1):
        i2 = PI*i2;
    if(j2<Nx/2+1):
        j2 = PI*j2;
    if(i2>=Nx/2+1):
        i2 = -PI*(Nx-i2);
    if(j2>=Nx/2+1):
        j2 = -PI*(Nx-j2);
    rta1 = 2*np.sin(dx*i2)/dx
    rta2 = 2*np.sin(dx*j2)/dx
    return 1.0/(np.power(i2,2)+np.power(j2,2));
    
outP = img_ft.copy()
outC = imgC.copy()
for k1 in range(Nx):
    for k2 in range(Nx):
        outP[k1,k2] = -4*PI*G*(img_ft[k1,k2])*calcK5(k1,k2);
        outC[k1,k2] = -4*PI*G*(imgC[k1,k2])*calcK5(k1,k2);



outC = fpotCreal+ 1j*fpotCimag
#outC *= (Nx)**2  #######################
#
#fast3d(np.abs(outP), "después python")
#fast3d(np.abs(outC), "después C")
#fast3d(np.abs(outP)-np.abs(outC), "diff C")


#Hasta aqué funciona consistentemente
############################################################

outMuerte= fft.fft2(potTheo) 
fast3d(np.abs(outP), "después python")
fast3d(np.abs(outC), "después C")
fast3d(np.abs(outMuerte), "Real ")
dMuerte3 = outMuerte[3,0]/outP[3,0]
dMuerte125 = outMuerte[125,0]/outP[125,0]
dMuerte20 = outMuerte[0,20]/outP[0,20]
dMuerte108 = outMuerte[0,108]/outP[0,108]
dMuerte3R = outMuerte[3,0]/(-4.0*PI*G*(imgC[3,0]))


#dout = outP - outC
#plt.figure()
#plt.imshow(np.real(testBackC*Nx**2-testBack))
#cbar = plt.colorbar()

def fastShow(image, title="none"):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.title(title)

potencialP = np.real(fft.ifft2(outP))
potencialC = np.real(fft.ifft2(outC))


pdiff = potencialP-potencialC
ddifft = testBackC - densidadTheo
#
#fastShow(potencialC, "potencial C")
#fastShow(potTheo, "teorico C")
#fastShow(potTheo - potencialC, title = "Potencial Teorico C - C")

#fastShow(potencialP, "potencial python")
#fastShow(potTheo, "teorico C")
#fastShow(potTheo - potencialP, title = "Potencial Teorico C - P")
#fastShow(pdiff, title = "Potencial Python - C")



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




outP = np.imag(outP)
outC = np.imag(outC)











