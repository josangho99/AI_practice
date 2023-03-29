#%%
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,4], [2,4], [3,8]])

w_range = np.linspace(start= - 10, stop = 10, num = 100)

sum_of_squares = []
for w in w_range:
    y_hat = w*data[:,0]+1
    e = data[:,1]-y_hat
    sum_of_squares.append(np.dot(e,np.transpose(e)))

plt.plot(w_range, sum_of_squares)

optimal_w = w_range[np.argmin(sum_of_squares)]
print(optimal_w)

plt.plot(optimal_w, np.min(sum_of_squares), 'ro')
plt.show()
# %%
