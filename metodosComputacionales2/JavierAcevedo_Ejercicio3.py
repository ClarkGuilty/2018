#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 07:09:33 2018

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt


def posterior(s):
    N = 10000
    x = np.linspace(0,1,N)
    prior = 1.0
    total = len(s)
    na = s.count('a')
    nb = s.count('b')
    print(total)
    prob = np.power(x,na)*np.power(1-x,nb)*prior
    moda = x[prob.argmax()]
    media = np.trapz(prob,x)
    prob = prob / media
    media = np.trapz(x*prob,x)
    sigma = np
    h = plt.figure()
    plt.plot(x,prob)
    plt.plot(x,likeli(x,media,sigma))
    plt.xlim(0,1)
    #plt.ylim(0,1.1)    
    sigma =  np.trapz(prob*x*x,x) - np.trapz(prob*x,x)**2
    plt.title(r'+prob = %.2f, Esperado = %.2f, $\sigma^2 = %.2f$' % (moda, media, sigma))
    plt.xlabel('Bias-weighting for a')
    plt.savefig(s+'.png')
    
    
def likeli(x, mean, sigma2):
    a = 1.0/(np.sqrt(2*np.pi*sigma2))
    b = -np.power(x-mean,2)/(2*sigma2)
    return a*np.exp(b)

posterior('ab')
    


