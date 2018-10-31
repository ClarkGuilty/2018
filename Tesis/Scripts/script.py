#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 19:47:26 2018

Script to generate some figures for my document.
@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte

##Galaxy rotation curves

R = 400
scale = 1*3000
rho0 = 1/300
Rmaxi = 10
M0 = 0.2

def rho(r):
    return rho0/(r/R)/(1+r/R)**2

def mass(rhor,x):
    return 2*np.pi*np.trapz(rhor*x,x)*scale

def rho2(r,Rmax,M0):
    rta = np.copy(r)
    rta[r > Rmax] = 0
    rta[r < Rmax] = M0
    return rta

def vorb(r,massa):
    return np.sqrt(massa/r)

r = np.linspace(0.1,160,250)
rhoa = rho(r)
rhob = rho2(r,Rmaxi, M0)
masa = np.zeros_like(rhoa)
masa2 = np.zeros_like(rhoa)
for i in range(1,len(masa)):
    masa[i] = mass(rhoa[0:i],r[0:i])
    #masa2[i]= mass2(r[i],100,100)
    masa2[i]= mass(rhob[0:i],r[0:i])
#masa2 = mass2(r,100,100)
plt.plot(r,vorb(r,masa), label = "Measured curves")
plt.plot(r,vorb(r,masa2), label = "Luminosity model")
plt.xlabel("Radius from the centre of the galaxy (arcmin)")
plt.ylabel("Orbital velocity (km/s)")
plt.legend()
plt.savefig("./imag/galaxyRotCurv.png", dpi = 500)
#print(mass(1,1))
#plt.plot(r,vorb(r))
#plt.plot(r,mass(r,1))
#plt.plot(r,np.sqrt(1.0*mass(r,1)/r))




