#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:20:58 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams.update({'text.usetex': True})

cKI = np.array([ 8.6e-4, 1.2e-1, 5.9e-1, 1.1, 2.8, 6.3])
tKI = np.array([ 1.0, 5, 8, 10, 15, 20])
print(tKI)

h = plt.figure(dpi=600)
plt.plot(tKI**3,cKI)
plt.scatter(tKI**3,cKI, color = 'r')
#plt.title(r'C vs T$^3$')
plt.title(r"\Huge{C vs $T^3$} \newline \small{}", fontsize=20)
tic = np.linspace(0,1.0,len(tKI))
#plt.xticks(tic, np.add(tKI,np.array(["r'^3'"]*len(tKI))))
plt.xticks(plt.xticks()[0], ["%.2f " % np.power(t,1/3.0) for t in plt.xticks()[0][1:]])
plt.ylabel(r'C (J $K^{-1}$ $mol^{-1}$)')
plt.xlabel(r'$T^3 (K^3)$')
#h.clf()
plt.savefig('punto2.png', bbox_inches="tight")

R = 8.1344

tD = np.power(12 * R*np.power(tKI,3)*np.pi**4/(5*cKI),1.0/3.0)
print(np.average(tD))
print(np.std(tD,ddof=1.0))

#
#Punto de Einstein

k = 1.3806e-23
x = np.linspace(0,2.5,1000)
x = x[1:]
cE = k*np.power(2*x,-2)*np.power(np.sinh(1.0/(2*x)),-2)
cons = np.array([k]*len(cE))
h1 = plt.figure(dpi=600)
plt.ylabel(r'C [J/K]')
plt.xlabel(r'$k_b T/\hbar w$')
plt.plot(x,cons, label = 'Dulong-Petit')
plt.plot(x,cE,label = 'Einstein')
plt.title("  C vs T modelo Einstein")
plt.legend()
plt.savefig('Einstein.png', bbox_inches="tight")