#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 06:49:57 2018

@author: Javier Alejandro Acevedo Barroso

"""
import numpy as np
import matplotlib.pyplot as plt

def heh(N, maxAge):
    Ntotal = 1000 #total de la población
    resolucion = 1000
    realMax = 100
    maxAges = []
    for age in range(1,realMax):
        for intentos in range(resolucion):            
            pob0 = np.random.randint(1,age+1, Ntotal)-1
            pMaxAge = np.max(pob0[0:N])

            if(pMaxAge == maxAge):
                maxAges += [age]

    maxAges = np.array(maxAges)

    plt.hist(maxAges,bins = 50, density = True)
    
    
    #print(distr)
    edadMaximaMasProbable = np.argmax(np.bincount(maxAges))
    edadMaximaEsperada = np.mean(maxAges)
    varianza = np.std(maxAges)
    plt.suptitle("Distr. de prob. con %d personas encontradas, y %d años de edad máxima" % (N,maxAge))
    plt.title(r'+prob = %d, Esperado = %.0f, $\sigma = %.2f$' % (edadMaximaMasProbable, edadMaximaEsperada, varianza))
    plt.ylabel('Densidad de probabilidad')
    plt.xlabel('Edad máxima de la población')
    plt.savefig('JavierAcevedo-ejercicio4.pdf')
    #for i in range(deno):
        

    
heh(2,40)