#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 06:51:22 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt

def alcance(v, theta):
    g = 9.8
    return v**2 * np.sin(2*theta)/g

n_puntos = 10000
d_obs = np.array([880, 795, 782, 976, 178])
sigma_d = 5.0

vmin = 90
vmax = 115 
#vmin y vmax se obtuvieron a partir de tanteos

def loglikelihood(y_obs, sigma_y_obs, vel, angle):
    dheh = d_obs -  alcance(vel, angle)
    dheh = dheh/sigma_y_obs
    dheh = -0.5 * np.sum(dheh**2)
    return dheh


def logprior(v, theta):
    p = -np.inf
    if v < vmax and v >vmin:
        p = 0.0
    return p

vinit = np.random.uniform(vmin,vmax)
theta_prior = np.random.uniform(0, np.pi/2.0, len(d_obs))

lista_v = [vinit]
logposterior = [loglikelihood(d_obs, sigma_d, lista_v[0], theta_prior) + logprior(lista_v[0], theta_prior)]

sigma_delta_v = 0.5

for i in range(1,n_puntos):
    propuesta_v  = lista_v[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_v)
    theta_prior_new = np.random.uniform(0, np.pi/2.0, len(d_obs))

    logposterior_viejo = loglikelihood(d_obs, sigma_d, lista_v[i-1], theta_prior_new) + logprior(lista_v[i-1], theta_prior_new)
    logposterior_nuevo = loglikelihood(d_obs, sigma_d, propuesta_v, theta_prior_new) + logprior(propuesta_v, theta_prior_new)

    r = min(1,np.exp(logposterior_nuevo-logposterior_viejo))
    alpha = np.random.random()
    if(alpha<r):
        lista_v.append(propuesta_v)
        logposterior.append(logposterior_nuevo)
    else:
        lista_v.append(lista_v[i-1])
        logposterior.append(logposterior_viejo)
lista_v = np.array(lista_v)
logposterior = np.array(logposterior)

mediaYo = lista_v.mean()
(values,counts) = np.unique(lista_v,return_counts=True)
ind=np.argmax(counts)
modaYo = values[ind]
desvYo = lista_v.std()


plt.xlabel(r"$v_0$ (km/s)")
plt.ylabel("Densidad de Probabilidad")
plt.grid()
hist, edges = np.histogram(lista_v, bins=np.arange(vmin,vmax,0.5), density=True)
plt.plot(edges[:-1], hist, label='Metropolis')
plt.suptitle('Velocidad que maximiza la probabilidad: {:.4}'.format(modaYo))
plt.title('Media: {:.4}   y desviacion estandar: {:.4}'.format(mediaYo, desvYo))
plt.legend()
plt.savefig("bayes.png")












