import numpy as np

A = np.array([[2,3],[4,5],[10,11]])
def transpose(A):
    for i in range(4):
        for j in range(3):
            if i!=j:
                A[i,j]=A[j,i]
transpose(A)
print(A)