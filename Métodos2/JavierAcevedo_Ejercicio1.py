#ex 1.1: number of seconds since the day I was born.
import datetime as dt
t0 = dt.date(1997,1,4)
tf = dt.date.today()
rta = tf - t0
daysToSec = 24*60*60
print("Ex 1.1 : number of seconds since the day I was born:")
print(rta.days*daysToSec)
print("")

#ex 1.2: golden ratio.
import math
rta = (1 + math.sqrt(5))*0.5
print("Ex 1.2: golden Ratio:")
print(rta)
print("\n")

print("Ex 2.01:")
lista = [1,2,3,4,5,6,7,8,9,10]
lista[int(len(lista)/2):]
print("\n")

print("Ex 2.02:")
l0 = [1,-1]
lista = 100*l0
lista
print("\n")


print("Ex 2.03:")
t0 = [0]
lista = 15*t0 + [1] + 15*t0
lista

print("Ex 2.04:")
a = [21, 48, 79, 60, 77, 15, 43, 90, 5, 49, 15, 52, 20, 70, 55, 4, 86, 49, 87, 59]
l0 = sorted(a)
if(len(a)%2 == 0):
    print(0.5*l0[len(a)//2] + 0.5 * l0[len(a)//2 + 1] )
else:
    print(l0[len(a)//2])
    

print("Ex: 3.01")
dic = {0: "a", 1:"e", 2:"i", 3:"o", 4:"u"}
dic






print("Ex 3.02:")
activities = {
    'Monday': {'study':4, 'sleep':8, 'party':0},
    'Tuesday': {'study':8, 'sleep':4, 'party':0},
    'Wednesday': {'study':8, 'sleep':4, 'party':0},
    'Thursday': {'study':4, 'sleep':4, 'party':4},
    'Friday': {'study':1, 'sleep':4, 'party':8},
}
def sumK(key):
    rta = 0
    for i in activities.keys():
        l0 = activities[i]
        rta += l0[key]
    return rta
        
total = {'study': sumK('study'), 'sleep' : sumK('sleep'), 'party': sumK('party')}
total


print("Ex 4.01:")
for i in reversed(range(0,6)):
    print(i)


print("Ex 4.02:")
from random import random as ran
a = 0
for i in range(100):
    a += ran()
print(a)
    
    
print("Ex 4.03:")
from random import random as ran
import statistics as sta
def calN(N):
    a = 0
    for i in range(N):
        a += ran()
    return a

lista = [calN(100)]
for i in range(999):
    lista = lista + [calN(100)]
print("La desviación estandar con N=100 es:")
print(sta.pstdev(lista))

eNe = 200
lista = [calN(eNe)]
for i in range(999):
    lista = lista + [calN(eNe)]
print("La desviación estandar con N=200 es:")
print(sta.pstdev(lista))

eNe = 300
lista = [calN(eNe)]
for i in range(999):
    lista = lista + [calN(eNe)]
print("La desviación estandar con N=300 es:")
print(sta.pstdev(lista))
print("\n")

print("Ex parte 2 1.1:")
from random import random as ran
def die():
    return int(ran()*6)+1

veces = 0
for i in range(1000):
    total = 0
    while(total <= 100):
        total +=die()
        veces +=1
print(veces/1000)
    
    
print("Ex 2.1")
def distance(a,b):
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i])**2
    return math.sqrt(total)
print(distance([0,0], [1,1]))
print(distance([1,5], [2,2]))
print(distance([0,1,2], [2,3,4]))


print("Ex 3.1")
class Circle:
    def __init__(self, radius):
        self.radius = radius #all attributes must be preceded by "self."
    def area(self):
        import math
        return math.pi * self.radius * self.radius
    def perimeter(self):
        import math
        return math.pi * 2.0 * self.radius	




print("Ex 3.2")
class Vector3D:
    def __init__(self, x, y, z):
        self.x = x 
        self.y = y 
        self.z = z 
    def dot(self, other):
        import math
        return self.x * other.x + self.y * other.y + self.z * other.z
    
v = Vector3D(2, 0, 1)
w = Vector3D(1, -1, 3)
print(v.dot(w))


print("Ex 4.1")
random.shuffle?
lista = list(range(10))
print(lista)
random.shuffle(lista)
print(lista)







