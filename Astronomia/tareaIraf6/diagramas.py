#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:34:47 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt

datos = np.loadtxt("outmatch.raw")

V = datos [:,3]
B = datos [:,5]
plt.scatter(B-V,V)
plt.gca().invert_yaxis()
plt.xlim(-0.1,2.5)
plt.xlabel('B-V')
plt.ylabel('V')
plt.title("Diagrama magnitud-color")
plt.savefig("VvsBV.png", dpi = 500)