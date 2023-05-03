#%%
import numpy as np
import matplotlib.pyplot as plt


X0 = np.random.rand(100)*10
y = 2*X0 + 1 + np.random.randn(100)

X = np.vstack((np.ones(100), X0))

W = np.ones((1,2))
W= W.reshape(-1, 1)

def model(w,x):
   return np.dot(w.T,x)

def loss(y, y_hat):
    return np.mean((y-y_hat)**2)

def OLS(x, y):
    w = np.ones((x.shape[0], 1))  # 가중치 벡터 초기화
    while True:
        y_hat = model(w, x)  # 예측값 계산
        gradient = np.mean((y_hat - y) * x, axis=1, keepdims=True)  # 경사하강법 계산
        w_new = w - 0.01 * gradient  # 학습률 0.01로 설정
        if np.allclose(w, w_new, rtol=1e-3):  # 가중치 수렴 확인
            break
        w = w_new
    return w

x_range = np.linspace(0, 10, 100)
x_const = np.vstack((np.ones(100), x_range))

y_init = model(W, x_const)
y_init = y_init.reshape(-1)

w_star = OLS(X,y)
y_hat = model(w_star, x_const)
y_hat = y_hat.reshape(-1)

plt.scatter(X0, y)
plt.plot(x_range, y_init, c='r', label = "initial")
plt.plot(x_range, y_hat, c='g', label= 'OLS')
plt.legend()
plt.show()

# %%
