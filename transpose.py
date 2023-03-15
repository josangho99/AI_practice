import numpy as np

a = np.array([[1,2],
              [3,4],
              [5,6]])

def transpose(A):
    AT=np.transpose(A)
    return AT

print(transpose(a))    
