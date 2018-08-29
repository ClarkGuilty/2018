#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 06:48:11 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt


x_obs = np.array([-2.0,1.3,0.4,5.0,0.1, -4.7, 3.0, -3.5,-1.1])
y_obs = np.array([ -1.931,   2.38,   1.88,  -24.22,   3.31, -21.9,  -5.18, -12.23,   0.822])
sigma_y_obs = ([ 2.63,  6.23, -1.461, 1.376, -4.72,  1.313, -4.886, -1.091,  0.8054])
plt.errorbar(x_obs, y_obs, yerr=sigma_y_obs, fmt='o')


realx = np.linspace(-5,5,100)

def model(x,q):
    a,b,c = q
    return a*x*x  + b * x + c

#q = (a,b,c)
def log_pdf_to_sample(q):
    rta = (y_obs - model(x_obs,q))/sigma_y_obs
    rta = rta.sum()
    return -0.5*rta*rta

def log_pdf_to_samplei(q,i):
    rta = (y_obs - model(x_obs,q))/sigma_y_obs
    return -0.5*rta[i]*rta[i]

def gradient_log_pdf_to_sample(q):
    h = 10e-2
    h0 = np.array([h,0,0])
    h1 = np.array([0,h,0])
    h2 = np.array([0,0,h])
    return 1.0/h*np.array((log_pdf_to_samplei(q+q*h0,0)/q[0] -log_pdf_to_samplei(q,0)/q[0] , log_pdf_to_samplei(q+q*h1,1)/q[1] -log_pdf_to_samplei(q,1)/q[1] , log_pdf_to_samplei(q+q*h2,2)/q[2] -log_pdf_to_samplei(q,2)/q[2] ))

def leapfrog(q,p, delta_t=5E-2, niter=20):
    q_new = q[:]
    p_new = p[:]
#    print(q_new)
#    print(p_new)
    for i in range(niter):
        p_new = p_new + 0.5 * delta_t * gradient_log_pdf_to_sample(q_new) #kick
        q_new = q_new + delta_t * p_new #drift
        p_new = p_new + 0.5 * delta_t * gradient_log_pdf_to_sample(q_new) #kick
#    print(q_new)
#    print(p_new)        
    return q_new, p_new


def H(q,p):
    K = 0.5 * p * p
    U = -log_pdf_to_sample(q)
    #U = ((model(x_obs,q)-y_obs)**2/sigma**2)
    return K.sum() + U


#h = plt.figure()
#plt.plot(realx,)
#print(H(heh,hehP,0.1))

N= 10000
lista_a = [np.random.normal()]
lista_pa = [np.random.normal()]
lista_b = [np.random.normal()]
lista_pb = [np.random.normal()]
lista_c = [np.random.normal()]
lista_pc = [np.random.normal()]
#logposterior = [loglikelihood(x_obs, y_obs, sigma_y_obs, lista_a[0], lista_b[0], lista_c[0]) + logprior(lista_a[0], lista_b[0], lista_c[0])]
sigma_delta_a = 0.1
sigma_delta_b = 0.1
sigma_delta_c = 0.1

sigma_delta_pa = 0.1
sigma_delta_pb = 0.1
sigma_delta_pc = 0.1

for i in range(1,N):
    #propuesta_a  = lista_a[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_a)
    #propuesta_b  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_b)
    #propuesta_c  = lista_b[i-1] + np.random.normal(loc=0.0, scale=sigma_delta_c)
    lista_pa.append(np.random.normal(loc=0.0, scale=sigma_delta_pa))
    lista_pb.append(np.random.normal(loc=0.0, scale=sigma_delta_pb))
    lista_pc.append(np.random.normal(loc=0.0, scale=sigma_delta_pc))
    q_new, p_new = leapfrog(np.array((lista_a[i-1],lista_b[i-1],lista_c[i-1])) , np.array((lista_pa[i-1],lista_pb[i-1],lista_pc[i-1])))
    #logposterior_viejo = loglikelihood(x_obs, y_obs, sigma_y_obs, lista_a[i-1], lista_b[i-1], lista_c[i-1]) + logprior(lista_a[i-1], lista_b[i-1], lista_c[i-1])
    #logposterior_nuevo = loglikelihood(x_obs, y_obs, sigma_y_obs, propuesta_a, propuesta_b, propuesta_c) + logprior(propuesta_a, propuesta_b,propuesta_c)
    E_new = H(q_new, p_new) # En lugar de evaluar la pdf se evalua la energia.
    E_old = H(np.array((lista_a[i-1], lista_b[i-1] , lista_c[i-1])), np.array((lista_pa[i-1], lista_pb[i-1] , lista_pc[i-1])))
    r = min(1.0,np.exp(-(E_new - E_old))) # Se comparan las dos energias
    #print(r)
    alpha = np.random.random()
    if(alpha<r):
        lista_a.append(q_new[0])
        lista_b.append(q_new[1])
        lista_c.append(q_new[2])
        #logposterior.append(logposterior_nuevo)
    else:
        lista_a.append(lista_a[i-1])
        lista_b.append(lista_b[i-1])
        lista_c.append(lista_c[i-1])
        #logposterior.append(logposterior_viejo)
lista_a = np.array(lista_a)
lista_b = np.array(lista_b)
lista_c = np.array(lista_c)

realx = np.linspace(-5,5,100)
#rta = (lista_a.argmax(),lista_b.argmax(),lista_c.argmax())
rta321 = (lista_a.mean(),lista_b.mean(),lista_c.mean())
plt.plot(realx, model(realx,rta321))
plt.title("a = %.3f  b = %.3f  c = %.3f" % (rta321))

h2 = plt.figure()
#plt.plot(lista_a, lista_a, alpha=0.5)
#plt.scatter(lista_a, lista_a, alpha=0.4, )
plt.hist(lista_a,bins = 100)
#plt.colorbar()
