#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:48:25 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt


k1 = 0.5
k2 = 1
m = 5
a = 1.5

J= 0.2
S = 1

x = np.linspace(-2,2,100)

def acusticPhonons1(k):
    return np.sqrt((k1+k2)/m + np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(k*a))/m)

def acusticPhonons2(k):
    return np.sqrt((k1+k2)/m - np.sqrt(k1**2 + k2**2 + 2*k1*k2*np.cos(k*a))/m)

def magnons(k):
    return 2*J*S*(1-np.cos(k*a))


plt.figure()

plt.title("Curvas de dispersión (2.d)",fontsize=20)
plt.xlabel("k",fontsize=20)
plt.ylabel('w',fontsize=20)
plt.plot(x,acusticPhonons1(x), label = 'Fonones ópticos')
plt.plot(x,acusticPhonons2(x), label = 'Fonones acústicos')
plt.plot(x,magnons(x), label = 'Magnones')
plt.legend(loc = 3)
plt.savefig('imprimirSolido1.png',dpi=300)


plt.figure()

def coth(x):
    return np.cosh(x)/np.sinh(x)

def eme(y,J):
    return (2.0*J+1.0)/(2.0*J)*coth((2.0*J+1.0)/(2.0*J)*y) - 1.0/(2.0*J)*coth(y/(2.0*J))





x = np.linspace(1e-4,25,100)
plt.title('Magnetización diferentes J (3.d)',fontsize=20)
plt.plot(x,eme(x,0.5), label = 'J = 1/2')
plt.plot(x,eme(x,1.5), label = 'J = 3/2')
plt.plot(x,eme(x,10000000), label = r'J = $\infty$')
plt.legend()
plt.xlabel('y',fontsize=20)
plt.ylabel(r'$M/M_s$',fontsize=20)
plt.savefig('imprimirSolido2.png',dpi=300)

plt.figure()

PARAMAGNETICO = 0
FERROMAGNETICO = -1
ANTIFERROMAGNETICO = 1

def invX(T,Tvalor,Tsigno):
    return T + Tsigno*Tvalor

T= np.linspace(0-0.3,1)
plt.plot(T,invX(T,0.1,PARAMAGNETICO),label ='Paramagnético')
plt.plot(T,invX(T,0.1,FERROMAGNETICO), label = 'Ferromagnético')
plt.plot(T,invX(T,0.1,ANTIFERROMAGNETICO), label = 'Antiferromagnético')
plt.legend()
plt.title('Inverso de la susceptibilidad',fontsize=20)
plt.xlabel('Temperatura',fontsize=20)
plt.ylabel(r'1 / $\chi$ ',fontsize=20)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.savefig('imprimirSolido3.png',dpi=300)














