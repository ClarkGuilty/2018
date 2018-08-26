#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:31:44 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt


#def integral_analitica():
#    return -np.cos(1)+np.cos(0)
#
#def integral_monte_carlo(N=100):
#    x = np.random.random(N)
#    return np.sum(np.sin(x))/N
#
#
#n_intentos = 10
#puntos = np.int_(np.logspace(1,5,n_intentos))
#diferencias = np.ones(n_intentos)
#for i in range(n_intentos):
#    a = integral_analitica()
#    b = integral_monte_carlo(N=puntos[i])
#    diferencias[i] =  (np.abs((a-b)/a))
#    
#    
#
#
#plt.plot(puntos, diferencias*100)
#plt.loglog()
#plt.xlabel("$N_{puntos}$")
#plt.ylabel("Diferencia porcentual Monte Carlo vs. Analitica")




N = 10000

def f(x):
    return np.sin(x)

def g(x):
    return x

def inv_cum(x):
    return np.sqrt(2*x)
#    return np.arccos(1-x)

def integral_analitica():
    return 1.0

def integral_monte_carlo(N=100):
    x = np.random.triangular(0,np.pi/2,np.pi/2,size=N)
#    x = g(np.random.rand(N)*np.pi/2.0)
#    x = inv_cum(np.random.random(N))
    return np.sum(f(x))/N *np.power(np.pi/2,2)/2

n_intentos = 30
puntos = np.int_(np.logspace(1,5,n_intentos))
diferencias = np.ones(n_intentos) 
for i in range(n_intentos):
    a = integral_analitica()
    b = integral_monte_carlo(N=puntos[i])
    diferencias[i] =  (np.abs((a-b)/a))
    
    
    
plt.plot(puntos, diferencias*100)
plt.loglog()
plt.xlabel("$N_{puntos}$")
plt.ylabel("Diferencia porcentual Monte Carlo vs. Analitica")


Np = 10000
plotx = np.linspace(0,np.pi/2,Np)
    
h = plt.figure()    
plt.hist(np.random.triangular(0,np.pi/2,np.pi/2,size=Np),density=True, bins= 100)
plt.hist(g(np.random.rand(Np)*np.pi/2.0), bins = 100, density = True, color ='r')
plt.hist(inv_cum(np.random.random(Np)), bins = 100, density = True, color ='g')
plt.plot(plotx, np.sin(plotx))
plt.plot(plotx, plotx)

    

#x = np.linspace(0,np.pi)
#    
#def f(x):
#    return np.sin(x)/2
#
#
#def samplear(sigma):
#    N = 100000
#    lista = [np.random.random()*np.pi]
#
#    for i in range(1,N):
#        propuesta  = lista[i-1] + np.random.normal(loc=0.0, scale=sigma)
#        r = min(1,f(propuesta)/f(lista[i-1]))
#        alpha = np.random.random()
#        if(alpha<r):
#            lista.append(propuesta)
#        else:
#            lista.append(lista[i-1])
#    return lista
#
#
#
#sigma1 = 0.001
#sigma2 = 1000.0
#
#plt.plot(x, f(x))
#_ = plt.hist(samplear(sigma1), density=True, bins=x, color = 'r')
#plt.hist(samplear(sigma2), density = True, bins = x, color = 'g')
#plt.hist(samplear(1), density = True, bins = x, color = 'y')
#
#print("Cuando el sigma es muy pequeño el tamaño del paso es demasiado pequeño para poder efectivamente recorrer la distribución y nunca sale de su región inicial.")
#print("Cuando el sigma es muy grande el tamaño del paso es demasiado grande y todos los candidatos nuevos terminan siendo no aptos para el sampleo, entonces nunca sale de su región inicial.")
    










#def dens(x, y):
#    return np.exp(-0.5*(x**2 + y**2/9 -x*y/12))
#
#x = np.linspace(-5,5)
#y = np.linspace(-5,5)
#N = 100000
#listax = [np.random.random()]
#listay = [np.random.random()]
##sigma_delta = 1.0
#sigma_delta = 0.8
#
#for i in range(1,N):
#    propuestax  = listax[i-1] + np.random.normal(loc=0.0, scale=sigma_delta)
#    propuestay  = listay[i-1] + np.random.normal(loc=0.0, scale=sigma_delta)
#    r = min(1,dens(propuestax,propuestay)/dens(listax[i-1],listay[i-1]))
#    alpha = np.random.random()
#    if(alpha<r):
#        listax.append(propuestax)
#        listay.append(propuestay)
#    else:
#        listax.append(listax[i-1])
#        listay.append(listay[i-1])
#
#
#
#
#
#x_grid, y_grid = np.meshgrid(x, y)
#z_grid = dens(x_grid, y_grid)
#fig, (ax0, ax1) = plt.subplots(1,2)
#
## grafica los puntos de la grid
#im = ax0.pcolormesh(x_grid, y_grid, z_grid)
#
## grafica el histograma bidimensional a partir de la lista de puntos
#_ = plt.hist2d(listax, listay, bins=40)
#plt.ylim([-5, 5])
#plt.xlim([-5, 5])
    
    
    
    
    