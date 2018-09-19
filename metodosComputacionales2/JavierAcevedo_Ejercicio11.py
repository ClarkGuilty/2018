#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 06:51:55 2018

@author: Javier Alejandro Acevedo Barroso

"""
import numpy as np
import matplotlib.pyplot as plt

#datos = pd.read_csv("years-lived-with-disability-vs-health-expenditure-per-capita.csv")
#data = datos[datos["Year"] ==2011]

datos = np.loadtxt("fitting.txt")
x = datos[:,0]
y = datos[:,1]
sigma = datos[:,2]


npol = 5

#plt.scatter(x,y)

def poly(x,params):
    n = len(params)
    rta = 0
    for j in range(n):
        rta += params[j] * x**j
    return rta

def loglikelihood(x_obs, y_obs, params):
    d = y_obs -  poly(x_obs, params)
    d = d/(sigma)
    #print(d)
    d = -0.5 * (d**2) -np.log(np.sqrt(2.0*np.pi*sigma**2))
    return np.sum(d)

def loglikelihood2(x_obs, y_obs, params):
    y_model = poly(x_obs, params)
    d = -0.5 * ((y_model - y_obs)/sigma)**2
    norm = np.sqrt(2.0 * np.pi * sigma **2)
    return np.sum(d - np.log(norm))

def logpriorParam(params):
    if (params <= 1.0  and params >= -1.0):
        area = (2)**2 
        p = np.log(1.0/area)
    else:
        p = -np.inf
    return p



npol = 1
N = 10000
evidencias = np.zeros(20)
evidencias2 = np.zeros(20)
mina = -1.0
maxa = 1.0
parametros = np.random.uniform(mina,maxa,(npol+1,N))
while(npol<20):    
    i = 1
    ytest= np.exp(loglikelihood(x,y, parametros[:,0]))
    like=[ytest]
    like2=[ytest]
    while(i<N):
        like.append(np.exp(loglikelihood(x,y,parametros[:,i])))
        like2.append(np.exp(loglikelihood2(x,y,parametros[:,i])))
        i+=1

    like = np.array(like)
    like2 = np.array(like2)
    evidencias[npol-1] = like.mean()
    evidencias2[npol-1] = like2.mean()
    print(like.mean())
    npol +=1
    parametros = np.random.uniform(mina,maxa,(npol+1,N))





enes = np.linspace(1,20,20)
plt.scatter(enes,evidencias)
plt.scatter(enes,evidencias2)



