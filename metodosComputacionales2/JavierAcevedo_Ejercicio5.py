#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 06:56:53 2018

@author: Javier Alejandro Acevedo Barroso

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.integrate as inte

def f(x,s):
    return np.exp(-0.5*x*x/s**2)/np.sqrt(2*np.pi*s*s)
    
def fbeta(x,a,b):
    return np.power(x,a-1)*np.power(1-x,b-1)*sp.gamma(a+b)/sp.gamma(a)/sp.gamma(b)

def finve(x,lam,C):
    rta = (np.exp(-lam/x))*(C/(x*x))
    return rta

N = 200000
x = np.arange(N,dtype=float)
permax = np.linspace(0.01,1,N)
s = 0.1
step = s*0.001
x[0] = 0.5
contador = 0
alpha = 2
beta = 5
for can in range(1,N,1):
    x_sig = np.random.uniform()
    prob = fbeta(x_sig,alpha,beta)/fbeta(x[can-1],alpha,beta)
    if(prob >= 1.0):
        x[can] = x_sig
        contador +=1
    else:
        if(np.random.uniform()<prob):
            x[can] = x_sig
            contador +=1
        else:
            x[can] = x[can-1]
      
        
h1 = plt.figure()
            
plt.hist(x,bins = 100, density=True)
#plt.plot(permax,f(permax,s))
plt.plot(permax,fbeta(permax,alpha,beta))
plt.suptitle(r'Distribución Beta con $\alpha$ = %.2f y $\beta$ %.2f' %(alpha,beta))
plt.title('La tasa de aceptación fue %.2f' %(contador/N))
plt.savefig('distBeta.pdf')



lamba = 5
C = 1
step = 0.
x[0] = 0.99
contador = 0

for can in range(1,N,1):
    x_sig = -1
    while(x_sig<=0 or x_sig>1):
        x_sig = step*(2*np.random.uniform()-1)+x[can-1]
    #print(x_sig)
    prob = finve(x_sig,lamba,C)/finve(x[can-1],lamba,C)
    #print('%f %f' % (x_sig, finve(x_sig,lamba,C)))
    if(prob >= 1.0):
        x[can] = x_sig
        contador +=1
    else:
        if(np.random.uniform()<prob):
            x[can] = x_sig
            contador +=1
        else:
            x[can] = x[can-1]
def funcion(x):
    return finve(x,lamba,C)

h2 = plt.figure()
plt.hist(x,bins=100,density = True)
y = finve(permax,lamba,C)
y = y/(inte.quad(funcion,0.001,1))[0]



plt.suptitle(r'Distribución inversa exponencial con $\alpha$ = %.2f y $\beta$ %.2f' %(alpha,beta))
plt.title('La tasa de aceptación fue %.2f' %(contador/N))
plt.plot(permax,y, color = 'r')
plt.savefig('invExpo.pdf')





            