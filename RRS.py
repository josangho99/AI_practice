#%%
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,4],[2,4],[3,8]])

plt.plot(data[:,0], data[:,1], 'ro')

x = np.linspace(start = 1, stop = 3, num = 3)
y = 1.7 * x + 2

print(y)

plt.plot(x, y, 'g-')

y_predict = data[:,1]
e = y - y_predict
sum_of_squares = np.dot(e,np.transpose(e))
print(sum_of_squares)

# %%
