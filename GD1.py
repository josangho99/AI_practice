#%%
import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3.3],[4.4],[5.5],[6.1],[6.3],[4.8],[9.7],[6.2],[7.9],[2.7],[7.2],[10.1],[5.3],[7.7],[3.1]])
y_train = np.array([[1.7],[1.9],[2.09],[2.1],[1.9],[1.3],[3.3],[2.5],[2.5],[1.1],[2.7],[3.4],[1.5],[2.4],[1.3]])

x_train_tenser = torch.FloatTensor(x_train)
y_train_tenser = torch.FloatTensor(y_train)

print(type(x_train_tenser))

# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 50
learning_rate = 0.01

#Linear Regression Model
model = torch.nn.Linear(input_size, output_size)

#Loss funtion, Optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

#Train the model
for epoch in range(num_epochs):
    optimizer.zero_grad()

    outputs = model(x_train_tenser)

    loss = criterion(outputs, y_train_tenser)

    loss.backward()
    optimizer.step()

    if(epoch+1) % 2 == 0 :
        print(f'Epoch : {epoch+1}, Loss: {loss.item():.2f}')
    
x_train_plot = torch.FloatTensor(x_train)
predicted = model(x_train_plot).detach().numpy()

plt.figure()
plt.plot(x_train, y_train, 'ro', label = 'data')
plt.plot(x_train, predicted, label = 'linear funtion')
plt.legend(loc='upper left')
plt.show()
# %%
