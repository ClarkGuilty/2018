#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 19:10:39 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt

par = {'Sirio': 379.21, 'Canopus' : 10.55, 'Alpha Centauri': 754.81, 'Arcturus': 88.83, 'Vega': 130.23,'Capella': 76.20, 'Rigel': 3.78,'Procyon': 284.56, 'Achernar': 23.39,'Betelgeuse': 4.51}
epar = {'Sirio': 1.58, 'Canopus' : 0.56, 'Alpha Centauri': 4.11, 'Arcturus': 0.54, 'Vega': 0.36,'Capella': 0.46, 'Rigel': 0.34,'Procyon': 1.26, 'Achernar': 0.57,'Betelgeuse': 0.80}
mag = {'Sirio': -1.46, 'Canopus' : -0.74, 'Alpha Centauri': -0.27, 'Arcturus': -0.05, 'Vega': 0.03,'Capella': 0.08, 'Rigel': 0.13,'Procyon': 0.34, 'Achernar': 0.46,'Betelgeuse': 0.50}
dist = np.array(list(par.values()))
dist = dist/1000
dist = 1.0/dist
names = list(par.keys())
dis = {names[x]: dist[x] for x in range(10)}
absMag = {names[x]: mag[names[x]] +5- 5*np.log10(dis[names[x]]) for x in range(10)}
copia = absMag.copy()

print("Estrella & Paralaje[mas] & Distancia[pc] & m & M \\\\")
for name in par.keys():
    print(r'%s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f & %.2f $\pm$ %.2f \\ \hline' % (name, par[name], epar[name], dis[name], (epar[name]/1000.0)/np.power(par[name]/1000, 2), mag[name], mag[name] +5-5*np.log10(1.0/(par[name]/1000.0) ),(epar[name]/1000.0)/np.power(par[name]/1000, 2)*5/dis[name]/np.log(10)))
    

#minAbs = 1000
#name = 'heh'
#for i in range(10):
#    minAbs = 1000
#    name = 'heh'
#    for left in absMag.keys():
#        if(minAbs > absMag[left]):
#            minAbs = absMag[left]
#            name = left
#    print(r'%s & %.2f $\pm$ %.2f & %.2f $\pm$ %.2f & %.2f & %.2f $\pm$ %.2f \\ \hline' 
#          % (name, par[name], epar[name], dis[name], (epar[name]/1000.0)/np.power(par[name]/1000, 2), 
#             mag[name], absMag.pop(name), (epar[name]/1000.0)/np.power(par[name]/1000, 2)*5/dis[name]/np.log(10) ))
                                                                                                        
        
        