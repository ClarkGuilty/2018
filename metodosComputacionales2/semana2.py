#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 23:17:28 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10e5
x = np.linspace(0,1,int(N))
triang = np.random.triangular(0,1,1,int(N))
plt.hist(triang, bins = 200, density = True)