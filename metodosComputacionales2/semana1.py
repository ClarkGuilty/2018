import math
def distance(a,b):
    dist = 0.0
    dif = 0.0
    for i in range(len(a)):
        dif = (a[i] - b[i])**2
        dist += dif
        dif = 0
    return math.sqrt(dist)


print(distance([0,0], [1,1]))
print(distance([1,5], [2,2]))
print(distance([0,1,2], [2,3,4]))
