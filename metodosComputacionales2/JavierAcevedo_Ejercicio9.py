#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 06:45:18 2018

@author: Javier Alejandro Acevedo Barroso

"""
import numpy as np
import matplotlib.pyplot as plt
#from plumbum.cmd import wget
import pandas as pd


datos = pd.read_csv("years-lived-with-disability-vs-health-expenditure-per-capita.csv")
data = datos[datos["Year"] ==2011]

x = np.array(datos["Health_expenditure_per_capita_PPP"])
x0 = x.copy()
y = np.array(datos["Years_Lived_With_Disability"])
y0 = y.copy()
popu = np.array(datos["Total_population_Gapminder)"])
popu0 = popu.copy()
y = y[np.logical_and( np.logical_and(np.isfinite(y0), np.isfinite(popu0)), np.isfinite(x0)  )]
popu = popu[np.logical_and( np.logical_and(np.isfinite(y0), np.isfinite(popu0)), np.isfinite(x0)  )]
x = x[np.logical_and( np.logical_and(np.isfinite(y0), np.isfinite(popu0)), np.isfinite(x0)  )]

#popu = popu[np.logical_not(np.isnan(x))]
#x = x[np.logical_not(np.isnan(x))]
#y = y[np.logical_not(np.isnan(x))]



plt.scatter(x,y, color = 'r')


def sigma(population):
    return population / np.max(popu)

def model1(x,a,b):
    return -a/x + b

def loglikelihood1(xobs,yobs,params):
    d = yobs -  model1(xobs, params[0], params[1])    
    d = d/sigma(popu)
    d = -0.5 * np.sum(d**2)
    return d

def logprior1(a, b):
    p = -np.inf
    if  a >0 and b >-10 and b<10:
        p = 0.0
    return p


N= 5000
lista_a = [np.random.random()]
lista_b = [np.random.random()]
lista_c = [np.random.random()]
logposterior = [loglikelihood1(x, y, (lista_a[0], lista_b[0])) + logprior1(lista_a[0], lista_b[0])]

sigma_delta_a = 0.2
sigma_delta_b = 1
sigma_delta_c = 1.0

for i in range(1,N):
    propuesta_a  = lista_a[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_a)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    #propuesta_c  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_c)

    logposterior_viejo = loglikelihood1(x, y, (lista_a[i-1], lista_b[i-1])) + logprior1(lista_a[i-1], lista_b[i-1])
    logposterior_nuevo = loglikelihood1(x, y , (propuesta_a, propuesta_b)) + logprior1(propuesta_a, propuesta_b)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_a.append(propuesta_a)
        lista_b.append(propuesta_b)
        #lista_c.append(propuesta_c)
        logposterior.append(logposterior_nuevo)
    else:
        lista_a.append(lista_a[i-1])
        lista_b.append(lista_b[i-1])
        #lista_c.append(lista_c[i-1])
        logposterior.append(logposterior_viejo)
lista_a = np.array(lista_a)
lista_b = np.array(lista_b)
lista_c = np.array(lista_c)
logposterior = np.array(logposterior)


realx = np.linspace(np.min(x),np.max(x),len(x))
#rta = (lista_a.argmax(),lista_b.argmax(),lista_c.argmax())
rta = (lista_a.mean(),lista_b.mean())
plt.plot(realx, model1(realx,rta[0],rta[1]), label ="-a/x + b , a = %.2f  b = %.2f" % (rta))
plt.title("Datos con los tres modelos")



##Segundo modelo
def model2(x,a,b):
    return a*np.log(x) + b

def loglikelihood2(xobs,yobs,params):
    d = yobs -  model2(xobs, params[0], params[1])    
    d = d/sigma(popu)
    d = -0.5 * np.sum(d**2)
    return d

def logprior2(a, b):
    p = -np.inf
    if  a >0 and b >-10 and b<10:
        p = 0.0
    return p


lista_a = [np.random.random()]
lista_b = [np.random.random()]
lista_c = [np.random.random()]
logposterior = [loglikelihood2(x, y, (lista_a[0], lista_b[0])) + logprior2(lista_a[0], lista_b[0])]

sigma_delta_a = 0.2
sigma_delta_b = 1
sigma_delta_c = 1.0

for i in range(1,N):
    propuesta_a  = lista_a[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_a)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    #propuesta_c  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_c)

    logposterior_viejo = loglikelihood2(x, y, (lista_a[i-1], lista_b[i-1])) + logprior2(lista_a[i-1], lista_b[i-1])
    logposterior_nuevo = loglikelihood2(x, y , (propuesta_a, propuesta_b)) + logprior2(propuesta_a, propuesta_b)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_a.append(propuesta_a)
        lista_b.append(propuesta_b)
        #lista_c.append(propuesta_c)
        logposterior.append(logposterior_nuevo)
    else:
        lista_a.append(lista_a[i-1])
        lista_b.append(lista_b[i-1])
        #lista_c.append(lista_c[i-1])
        logposterior.append(logposterior_viejo)
lista_a = np.array(lista_a)
lista_b = np.array(lista_b)
lista_c = np.array(lista_c)
logposterior = np.array(logposterior)



realx = np.linspace(np.min(x),np.max(x),len(x))
#rta = (lista_a.argmax(),lista_b.argmax(),lista_c.argmax())
rta = (lista_a.mean(),lista_b.mean())
plt.plot(realx, model2(realx,rta[0],rta[1]), label = "a * ln(x) + b , a = %.2f  b = %.2f" % (rta))




##Tercer modelo
def model3(x,a,b,c):
    #return a*x+x + b*x+c
    #return np.power(x,a)*np.power((b-x),c)
    return a*np.log(x)+x**b+c

def loglikelihood3(xobs,yobs,params):
    d = yobs -  model3(xobs, params[0], params[1], params[2])    
    d = d/sigma(popu)
    d = -0.5 * np.sum(d**2)
    return d

def logprior3(a, b, c):
    p = -np.inf
    if  a > 0 and b >-10 and b<10 and c<100 and c>-100:
        p = 0.0
    return p


lista_a = [np.random.random()]
lista_b = [np.random.random()]
lista_c = [np.random.random()]
logposterior = [loglikelihood3(x, y, (lista_a[0], lista_b[0],lista_c[0])) + logprior3(lista_a[0], lista_b[0],lista_c[0])]

sigma_delta_a = 0.2
sigma_delta_b = 0.1
sigma_delta_c = 1.0

for i in range(1,N):
    propuesta_a  = lista_a[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_a)
    propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    propuesta_c  = lista_c[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_c)

    logposterior_viejo = loglikelihood3(x, y, (lista_a[i-1], lista_b[i-1],lista_c[i-1])) + logprior3(lista_a[i-1], lista_b[i-1],lista_c[i-1])
    logposterior_nuevo = loglikelihood3(x, y , (propuesta_a, propuesta_b, propuesta_c)) + logprior3(propuesta_a, propuesta_b, propuesta_c)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_a.append(propuesta_a)
        lista_b.append(propuesta_b)
        lista_c.append(propuesta_c)
        logposterior.append(logposterior_nuevo)
    else:
        lista_a.append(lista_a[i-1])
        lista_b.append(lista_b[i-1])
        lista_c.append(lista_c[i-1])
        logposterior.append(logposterior_viejo)
lista_a = np.array(lista_a)
lista_b = np.array(lista_b)
lista_c = np.array(lista_c)
logposterior = np.array(logposterior)



realx = np.linspace(np.min(x),np.max(x),len(x))
#rta = (lista_a.argmax(),lista_b.argmax(),lista_c.argmax())
rta = (lista_a.mean(),lista_b.mean(), lista_c.mean())
plt.plot(realx, model3(realx,rta[0],rta[1],rta[2]), label = "aln(x) + x**b + c , a = %.2f  b = %.2f c = %.2f" % (rta))
plt.legend()
plt.savefig("tresModelos.png")








