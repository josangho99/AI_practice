import numpy as np

a = np.array([[1],
              [2],
              [3]])
b= np.array([[3,2,1]])

def dot(A,B):
    result = np.dot(A,B) # np.dot 행렬곱
    return result

print(dot(a,b))
print(dot(b,a))
print(a@b) # A@B 내적
print(b@a)
print(a*b) # A*B, 원소 곱
print(b*a)

print(a.shape) #(row,column)
print(a.shape[0]) # row
print(a.shape[1]) # column