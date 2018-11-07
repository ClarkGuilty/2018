#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 20:08:03 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_table('MASTER.raw')



raw1 = np.loadtxt('MASTER.raw', usecols=1)