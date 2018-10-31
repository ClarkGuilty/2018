#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:18:38 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

puntos = np.array([[5,13.36],[15,39.36], [5,13.16], [15,38.9] , [25,88.42]      ])
tiempo = np.arange(1,75)
Nt = puntos[:,0]
t = puntos[:,1]


reg = LinearRegression().fit(Nt.reshape(-1,1),t)
#print(reg.coef_[0])
#print(reg.intercept_ )

def demora(Nt):
    return reg.coef_[0]*Nt+reg.intercept_

#plt.plot(tiempo,demora(tiempo))
#plt.scatter(Nt,t)
print(demora(50))