import numpy as np

a = np.array([[1],
              [2],
              [3]])
b= np.array([[3,2,1]])

def dot(A,B):
    result = np.dot(A,B)
    return result

print(dot(a,b))
print(dot(b,a))
print(a@b)
print(b@a)
print(a*b)
print(b*a)