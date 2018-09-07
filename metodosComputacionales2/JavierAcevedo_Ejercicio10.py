#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 06:47:56 2018

@author: Javier Alejandro Acevedo Barroso

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


datos = pd.read_csv("years-lived-with-disability-vs-health-expenditure-per-capita.csv")
data = datos[datos["Year"] ==2011]

x = np.array(datos["Health_expenditure_per_capita_PPP"])
x0 = x.copy()
y = np.array(datos["Years_Lived_With_Disability"])
y0 = y.copy()
popu = np.array(datos["Total_population_Gapminder)"])
popu0 = popu.copy()
y = y[np.logical_and( np.logical_and(np.isfinite(y0), np.isfinite(popu0)), np.isfinite(x0)  )]
popu = popu[np.logical_and( np.logical_and(np.isfinite(y0), np.isfinite(popu0)), np.isfinite(x0)  )]
x = x[np.logical_and( np.logical_and(np.isfinite(y0), np.isfinite(popu0)), np.isfinite(x0)  )]



#plt.scatter(x,y, color = 'r')


def sigma(population):
    return population / np.max(popu)

def model1(x,param):
    return param[0]*np.log(x)+param[1]

#def loglikelihood1(xobs,yobs,params):
#    d = yobs -  model1(xobs, params[0], params[1])    
#    d = d/sigma(popu)
#    d = -0.5 * np.sum(d**2)
#    return d


#def loglikelihood(x_obs, y_obs, param):
#    y_model = model1(x_obs, param)
#    p = y_model * np.exp(-(y_model/y_obs))# gamma con k=2 https://en.wikipedia.org/wiki/Gamma_distribution
#    p = p/(y_obs**2)
#    p = np.log(p)
#    return np.sum(p)
    
def loglikelihood(x_obs, y_obs, y_model):
    p = y_model * np.exp(-(y_model/y_obs))# gamma con k=2 https://en.wikipedia.org/wiki/Gamma_distribution
    p = p/(y_obs**2)
    p = np.log(p)
    return np.sum(p)


def logprior1(a, b):
    p = -np.inf
    if  a >0 and b >-10 and b<10:
        p = 0.0
    return p

def logpriorParam(param):
    if param.all() > 0 and param.all() < 10:
        area = (10)**2 
        p = np.log(1.0/area)
    else:
        p = -np.inf
    return p

latest1 = np.random.uniform(0,20,len(y))
lbtest1 = np.random.uniform(0,20,len(x))


i = 1
ytest1 = model1(x, [latest1[i],lbtest1[i]])
like1=[ytest1]
while(i<len(latest1)):
    like1.append(model1(x,[latest1[i],lbtest1[i]]))
    i+=1

like1 = np.array(like1)
likel1 = like1.mean()


def model2(x,param):
    return -param[0]/x + param[1]



def logpriorParam2(param):
    if param[0] > 0 and param.all() < 10:
        area = (10)**2 
        p = np.log(1.0/area)
    else:
        p = -np.inf
    return p

latest2 = np.random.uniform(0,20,len(y))
lbtest2 = np.random.uniform(0,20,len(x))


i = 1
ytest2 = model2(x, [latest2[i],lbtest2[i]])
like2=[ytest2]
while(i<len(latest2)):
    like2.append(model2(x,[latest2[i],lbtest2[i]]))
    i+=1

like2 = np.array(like2)
likel2 = like2.mean()



def model3(x,param):
    return param[0]*np.log(x)+x**param[1]+param[2]



def logpriorParam3(param):
    if param[0] > 0 and param[1] >-10 and param[1]<10 and param[2]<100 and param[2] >-100:
        area = (10)**2 
        p = np.log(1.0/area)
    else:
        p = -np.inf
    return p

latest3 = np.random.uniform(0,20,len(y))
lbtest3 = np.random.uniform(0,20,len(x))
lctest3 = np.random.uniform(0,20,len(x))


i = 1
ytest3= model3(x, [latest3[i],lbtest3[i],lctest3[i]])
like3=[ytest3]
while(i<len(latest3)):
    like3.append(model3(x,[latest3[i],lbtest3[i],lctest3[i]]))
    i+=1

like3 = np.array(like3)
likel3 = like3.mean()

F12 = likel1 / likel2
F23 = likel2 / likel3
F31 = likel3 / likel1

print("F12 = %f , F23 = %f , F31 = %f " %(F12,F23,F31))