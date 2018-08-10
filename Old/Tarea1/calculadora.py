# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 20:21:23 2018

@author: Javier Alejandro Acevedo Barroso
"""
from numpy import sqrt
from fractions import Fraction

def jm(J,M):
    print "J = "+str(J)+ " ; M = "+str(M)+ " ; rta = "+ str( Fraction((J*(J+1) - M*(M-1))).limit_denominator())
#    return J*(J+1)- M*(M-1)
    
jm(3/2.,-1/2.)
jm(2,-1)
jm(1/2.,0.5)
jm(2,0)
