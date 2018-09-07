#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:37:13 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt

Nx = 128
x = np.linspace(-1,1,Nx)
def fun(x):
    return np.sin(0.5*np.pi*x)

plt.plot(x,fun(x)**2)