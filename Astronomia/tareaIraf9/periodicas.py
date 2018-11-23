#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 21:58:34 2018

@author: Javier Alejandro Acevedo Barroso
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit




if(len(sys.argv)>2):
    print("El uso correcto es {s} threshold".format(sys.argv[0]))
    exit(0)
threshold = (float) (sys.argv[1])


medidas = np.loadtxt("rta.txt", skiprows=1, usecols=[4,5] )
#medidas = medidas[~np.isnan(medi]
mag = medidas[:,0]
emag = medidas[:,1]

ii = np.logical_and(np.isfinite(mag),np.isfinite(emag))
mag = mag[ii]
emag = emag[ii]

plt.figure(figsize=(12,7))
plt.scatter(mag,emag, alpha = 0.5, color = 'indigo')



def model1(x,a,b,c):
    return a*np.exp(b*x) + c

def model2(x,a,b,c):
    return a*np.log(b*x)+c


modelos = ['cuadratico', 'exponencial', 'logaritmico']

x = np.linspace(np.min(mag), np.max(mag), len(mag))


params1 = curve_fit(model1, xdata=mag, ydata= emag,     p0=(-1,0,1) )[0]
plt.plot(x,2*model1(x,params1[0], params1[1], params1[2]),label=modelos[1])


params2 = curve_fit(model2,xdata= mag,ydata= emag)[0]
plt.plot(x,2*model2(x,params2[0], params2[1], params2[2]))

K = range(1,20)
#K = [17,15,13]
for grade in K:
    fit = np.polynomial.polynomial.polyfit(mag,emag,grade)
    y = np.polynomial.polynomial.polyval(x,fit)
    plt.plot(x,2.3*y, label='grado '+str(grade))
plt.xlabel("Magnitud",fontsize=20)
plt.ylabel("Dispersión",fontsize=20)
plt.legend()




plt.figure(figsize=(12,7))
plt.scatter(mag,emag, alpha = 0.5, color = 'indigo')

#threshold = 3
K = [13]
for grade in K:
    fit = np.polynomial.polynomial.polyfit(mag,emag,grade)
    y = np.polynomial.polynomial.polyval(x,fit)
    ii = emag > np.polynomial.polynomial.polyval(mag,fit)*threshold
    plt.scatter(mag[ii],emag[ii], alpha = 0.5, color = "xkcd:red")
    plt.scatter(mag[~ii],emag[~ii], alpha = 0.5, color = "blue")
    plt.plot(x,threshold*y, label='grado '+str(grade), color='lime')
plt.xlabel("Magnitud",fontsize=20)
plt.ylabel("Dispersión",fontsize=20)
plt.savefig("tareaFN/threshold.png")
plt.legend()

datos = pd.read_csv('rta.txt', sep='      ',skiprows=1,header=None,  names=['ID', 'n','mean','sigma','rob_mean','rob_sigma','max','min'])
nombres = datos['ID']

ii = emag > np.polynomial.polynomial.polyval(mag,fit)*threshold


candidatas = nombres[ii]
with open('candidatas.txt', 'w+') as f:
    for filename in candidatas:
        print(filename, file=f)
        
print("Se encontraron ", len(nombres[ii])," candidatas a partir del Threshold.")












