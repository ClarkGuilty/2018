#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 09:25:50 2018

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt


pans =  open('tareaFN/periodoCandidatasAIRAF.txt', 'w+')  
ans =  open('tareaFN/candidatasAIRAF.txt', 'w+')  

lines = [line.rstrip('\n') for line in open("tareaFN/names.txt")]


for datfile in lines:
    datos = np.loadtxt("tareaFN/"+datfile+".rta", skiprows=4)
    ii = (datos[:,1] > 1.999)
    periodos = datos[ii]
    numeroArchivos = len(periodos)
    for periodo in periodos[:,1]:
        print(datfile, periodo, file=pans)
    if(len(periodos[:,1]) > 0):
        print(datfile, file=ans)
    
    
