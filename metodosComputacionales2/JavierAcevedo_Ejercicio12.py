#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 06:47:37 2018

@author: Javier Alejandro Acevedo Barroso

"""
import numpy as np
import matplotlib.pyplot as plt

datos = np.loadtxt("fitting.txt")
x = datos[:,0]
y = datos[:,1]
sigma = datos[:,2]

xtrain = x[0:10]
ytrain = y[0:10]
xtest = x[10:20]
ytest = y[10:20]
strain = sigma[0:10]
stest = sigma[10:20]

xreal = np.linspace(min(x),max(x),int(len(x)/2))

#plt.scatter(x[0:10], y[0:10])
#plt.scatter(x[10:20], y[10:20])


npol = 5

#plt.scatter(x,y)

def poly(x,params):
    n = len(params)
    rta = 0
    for j in range(n):
        rta += params[j] * x**j
    return rta


def loglikelihood(x_obs, y_obs, s_ob, params):
    y_model = poly(x_obs, params)
    d = -0.5 * ((y_model - y_obs)/s_ob)**2
    return np.sum(d)


def logprior(params):
    if( (params > 1.0).any() and (params < -1.0).any() ):
        return -np.inf;
    return 0.0

def SME(x_obs,y_obs, sigma, params):
    d = (y_obs - poly(x_obs,params))/sigma
    return (1.0/len(params)* (np.sum(d**2)))

def best(listaParams):
    rta = np.zeros_like(listaParams[:,0])
    for k in range(len(rta)):
        rta[k] = np.mean(listaParams[k,:])
    return rta
    
def entrenar(grado=5,N = 10000):
#    n = len(params)
    n = grado +1
#    print(n)
#    print("satan")
#    n = int(n)
    propuestas = np.zeros(n)
    listas = np.zeros((n, N))
    logposterior = np.zeros(N)
    listas[:,0] = np.random.random(len(listas[:,0]))
    for i in range(1,N):
        propuestas  = listas[:,i-1] + np.random.normal(loc=0.0, scale=0.3, size = n)
        #print(propuestas.shape)
        logposterior_viejo = loglikelihood(xtrain, ytrain, strain, listas[:,i-1]) + logprior(listas[:,i-1])
        logposterior_nuevo = loglikelihood(xtrain, ytrain, strain, propuestas) + logprior(propuestas)
#        print(logposterior_nuevo, logposterior_viejo)
#        print(propuestas)
        r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
        alpha = np.random.random()
        if(alpha<r):
            listas[:, i] = propuestas
#        print(listas[:,i] - propuestas)
            logposterior[i] = logposterior_nuevo
        else:
            listas[:, i] = listas[:,i-1]
#        print(listas[:,i] - listas[:,i-1])
            logposterior[i] = logposterior_viejo
#    print(listas)
    return listas
parametrosTodos = np.zeros((11,10)) #
fit2 = entrenar(grado = 5)
pa2 = best(fit2)
#for i in range(10):
#    parametrosTodos[i, :] = best(entrenar(n=i))
        

smeTrain = np.zeros((10))
smeTest = np.zeros((10))
d = np.linspace(1,10,10)
for i in range(len(d)):
    smeTrain[i] = SME(xtrain,ytrain,strain,best(entrenar(grado = i)))
    smeTest[i] = SME(xtest,ytest,stest,best(entrenar(grado = i)))
plt.plot(d, smeTrain, label = "Entrenamiento")
plt.plot(d, smeTest, label = "prueba")
plt.legend()
plt.savefig("sme.pdf")
#lista_m = np.array(lista_m)
#lista_b = np.array(lista_b)
#logposterior = np.array(logposterior)
#pa2 = np.zeros(3)
#plt.scatter(xtrain,ytrain)
#plt.plot(xreal,poly(xreal,pa2))
#grado 2



##Training
    
