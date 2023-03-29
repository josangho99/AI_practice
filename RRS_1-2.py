#%%
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1, 4], [2, 4], [3, 8]])

w_0 = np.linspace(start=-10, stop=10, num=100)
w_1 = np.linspace(start=-10, stop=10, num=100)

sum_of_squares = np.zeros((100, 100))

for i in range(100):
    for j in range(100):
        y_hat = w_0[i] + w_1[j]*data[:,0]
        e = data[:,1] - y_hat
        sum_of_squares[i,j] = np.sum(np.dot(e,np.transpose(e)))

plt.contour(w_0, w_1, sum_of_squares, levels=100)
plt.colorbar()

basepoint_w_0 = 1
basepoint_w_1 = 2
plt.scatter(basepoint_w_0, basepoint_w_1, c = 'g')

optimal_w_0 = w_0[np.argmin(sum_of_squares) //100]
optimal_w_1 = w_1[np.argmin(sum_of_squares) % 100]
plt.scatter(optimal_w_0, optimal_w_1, c='r')

plt.show()
# %%
