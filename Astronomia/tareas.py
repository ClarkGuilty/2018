#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:50:00 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np

cAUaMetros = 149597900000 #Metros en una Astronomical Unit.


#Tarea 1 

cSolarConstant = 1361   # constante solar en  W/m^2
solar = cSolarConstant * cAUaMetros**2 #constante solar en W/AU^2
irradianciaSol = 4 * np.pi * solar #Irradiancia usando la constante solar en W/AU^2


#Semieje mayor en unidades astronómicas.
distancias = {"Mercurio":0.39,'Venus':0.723,'Tierra': 1.0, 'Marte': 1.524,'Júpiter': 5.203, 'Saturno': 9.539, 'Urano': 19.18,'Neptuno': 30.06}

for planet in list(distancias.keys()):
    print("La constante solar en %s es %.2f [W/m^2] y %.0f [W/AU^2] a %.2f [AU]" % (planet,irradianciaSol/(4*np.pi* distancias[planet]**2*cAUaMetros**2),irradianciaSol/(4*np.pi* distancias[planet]**2), distancias[planet] ))