#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 02:50:53 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt


im011 = np.loadtxt("datos011.txt")
im010 = np.loadtxt("datos010.txt")

dx = im010[:,0] - im011[:,0]
dy = im010[:,1] - im011[:,1]

print("%.2f %.2f" %(np.median(dx),np.median(dy)))