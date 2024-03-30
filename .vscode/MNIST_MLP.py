#%%#
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #오류 무시하는 코드

# 데이터 불러오기 및 전처리
np.random.seed(777)

# hyperparameters
training_epochs = 300 # 큰 값으로 잡음(vaildation loss 떨어지면 자동 종료되기 때문에)
batch_size = 32
learning_rate = 0.01
input_size = 784    # 28x28 image       # 고정된 값 (이미지 크기)
hidden_size1 = 300  # 임의의 값          # 임의의 값 (hidden layer의 노드 개수)
hidden_size2 = 150
output_size = 10    # 0~9               # 고정된 값 (분류할 클래스 개수)

# 데이터 로드 및 분할
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리 (예: RGB 0-255 범위의 픽셀값을 0-1 범위로 정규화)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 레이블 전처리 (원-핫 인코딩)
train_labels = tf.one_hot(train_labels, depth=output_size)
test_labels = tf.one_hot(test_labels, depth=output_size)

# 데이터셋 분리
validation_size = 0.1
train_size = int ((1.0- validation_size)* len(train_images))
train_image, valid_image = train_images[:train_size], train_images[train_size:]
train_labels, valid_labels = train_labels[:train_size], train_labels[train_size:]

#행렬 연산을 위해 reshape
test_images = test_images.reshape(-1, 784)

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    c = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_output = exp_x / sum_exp_x
    return softmax_output

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def batch_cross_entropy(y_true, y_pred):
        epsilon = 1e-7  # 0으로 나누는 것을 방지하기 위한 작은 값

        # 예측값이 0 또는 1에 가까워지면 log(0)이나 log(1)은 무한대가 되므로 작은 값을 더해줌
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

        # 배치 크기 계산
        batch_size = y_true.shape[0]

        # 크로스 엔트로피 계산
        ce = -np.sum(y_true * np.log(y_pred)) / batch_size

        return ce

class MINSTMLP:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.epsilon = 1e-7
        
        # 가중치 초기화
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * 0.01
        self.b1 = np.zeros(self.hidden_size1)
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.01
        self.b2 = np.zeros(self.hidden_size2)
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.01
        self.b3 = np.zeros(self.output_size)

        #Adagrad-동일한 크기이며 원소가 모두 0인 행렬 생성
        self.G_W3 = np.zeros_like(self.W3)
        self.G_b3 = np.zeros_like(self.b3)
        self.G_W2 = np.zeros_like(self.W2)
        self.G_b2 = np.zeros_like(self.b2)
        self.G_W1 = np.zeros_like(self.W1)
        self.G_b1 = np.zeros_like(self.b1)
    
    #순전파
    def forward(self, X):

        X = np.reshape(X, (X.shape[0], -1))
        # 첫 번째 은닉층 순전파
        self.hidden1 = np.dot(X, self.W1) + self.b1
        self.hidden1_output = relu(self.hidden1)
        
        # 두 번째 은닉층 순전파
        self.hidden2 = np.dot(self.hidden1_output, self.W2) + self.b2
        self.hidden2_output = relu(self.hidden2)
        
        # 출력층 순전파
        self.output = np.dot(self.hidden2_output, self.W3) + self.b3
        self.output_prob = softmax(self.output)
        
        return self.output_prob
    
    #예측
    def predict(self, X):
        predictions = self.forward(X)
        return predictions
    
    #정확도 계산
    def calculate_accuracy(self, X, y):
        predictions = self.predict(X)
        predictions = np.argmax(predictions, axis=1) #퍼센트가 가장 높은 것을 one-hot-encoding() 하여 한 개의 숫자만 추출
        true = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true)
        return accuracy
    
    #loss 계산
    def loss(self, y_true, y_pred):
        loss = batch_cross_entropy(y_true, y_pred)
        return loss
    
    #역전파
    def backpropagation(self, X, y_true, y_pred):
        # Output layer의 gradient 계산
        output_error = y_pred - y_true
        output_gradient = np.dot(self.hidden2_output.T, output_error) / X.shape[0]

        # Hidden layer 2의 gradient 계산
        hidden2_error = np.dot(output_error, self.W3.T)
        hidden2_gradient = np.dot(self.hidden1_output.T, hidden2_error) / X.shape[0]

        # Hidden layer 1의 gradient 계산
        hidden1_error = np.dot(hidden2_error, self.W2.T)
        hidden1_gradient = np.dot(X.reshape(-1, self.input_size).T, hidden1_error) / X.shape[0]
        
        #Adagrad 구현
        self.G_W3 += np.square(output_gradient)
        self.W3 -= (self.learning_rate / (np.sqrt(self.G_W3) + self.epsilon)) * output_gradient

        self.G_b3 += np.square(np.mean(output_error, axis=0))
        self.b3 -= (self.learning_rate / (np.sqrt(self.G_b3) + self.epsilon)) * np.mean(output_error, axis=0)

        self.G_W2 += np.square(hidden2_gradient)
        self.W2 -= (self.learning_rate / (np.sqrt(self.G_W2) + self.epsilon)) * hidden2_gradient

        self.G_b2 += np.square(np.mean(hidden2_error, axis=0))
        self.b2 -= (self.learning_rate / (np.sqrt(self.G_b2) + self.epsilon)) * np.mean(hidden2_error, axis=0)

        self.G_W1 += np.square(hidden1_gradient)
        self.W1 -= (self.learning_rate / (np.sqrt(self.G_W1) + self.epsilon)) * hidden1_gradient

        self.G_b1 += np.square(np.mean(hidden1_error, axis=0))
        self.b1 -= (self.learning_rate / (np.sqrt(self.G_b1) + self.epsilon)) * np.mean(hidden1_error, axis=0)

    #트레이닝
    def train(self, train_images, train_labels, valid_images, valid_labels):
        num_batches = len(train_images) // self.batch_size # train 데이터의 미니배치 개수
        loss_history = []
        train_acc_history = []
        valid_acc_history = []
        val_loss = [] 
        patience = 3
        best_val_loss = float('inf')
        num_valid_batches = len(valid_images) // self.batch_size  # 검증 데이터의 미니배치 개수
        total_valid_loss = 0  # 전체 검증 손실 초기화
        valid_accuracy = 0  # 검증 정확도 초기화

        for epoch in range(1, self.training_epochs + 1):
            total_loss = 0.0

            # 미니배치 학습
            for batch in range(num_batches):
                # 미니배치 데이터 가져오기
                batch_images = train_images[batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_labels = train_labels[batch * self.batch_size: (batch + 1) * self.batch_size]

                # 순방향 전파 수행
                y_pred = self.forward(batch_images)

                # 손실 계산 및 출력
                batch_loss = self.loss(batch_labels, y_pred)
                total_loss += batch_loss

                # 역전파 수행
                self.backpropagation(batch_images, batch_labels, y_pred)
                train_loss = total_loss / num_batches

            # 학습 과정 출력
            print(f"Epoch {epoch}/{self.training_epochs}, Loss: {(train_loss):.4f}%")
            train_accuracy = self.calculate_accuracy(train_images, train_labels)
            print(f"Training Accuracy: {train_accuracy*100:.4f}%")

            total_valid_loss = 0  # 검증 손실 초기화
            for valid_batch in range(num_valid_batches):
                # 검증 데이터의 미니배치 가져오기
                batch_valid_images = valid_images[valid_batch * self.batch_size: (valid_batch + 1) * self.batch_size]
                batch_valid_labels = valid_labels[valid_batch * self.batch_size: (valid_batch + 1) * self.batch_size]

                # 순방향 전파 수행하여 예측값 계산
                y_pred_valid = self.forward(batch_valid_images)

                # 검증 손실 계산
                batch_valid_loss = self.loss(batch_valid_labels, y_pred_valid)
                total_valid_loss += batch_valid_loss

            valid_loss = total_valid_loss / num_valid_batches # 배치 개수로 나누어 loss 평균 구함
            val_loss.append(valid_loss)
            print(f"Validation Loss: {valid_loss:.4f}")

            # Early stopping 검사
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                patience = 3
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping. Best Validation Loss: {:.4f}".format(best_val_loss))
                    break

            loss_history.append(train_loss)
            train_acc_history.append(train_accuracy)
            valid_accuracy = self.calculate_accuracy(valid_images, valid_labels)
            valid_acc_history.append(valid_accuracy)
            print(f"Validation Accuracy: {valid_accuracy*100:.4f}%")

        # 그래프 그리기
        #Loss curve
        plt.plot(loss_history, label="train")
        plt.plot(val_loss, label="valid")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss curve")
        plt.show()

        plt.figure()
        plt.plot(train_acc_history, label='Training Accuracy')
        plt.plot(valid_acc_history, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

#mlp 변수로 초기화
mlp = MINSTMLP(input_size, hidden_size1, hidden_size2, output_size, learning_rate)

#train
mlp.train(train_image, train_labels, valid_image, valid_labels)
test_predictions = mlp.predict(test_images)

#test
test_accuracy = mlp.calculate_accuracy(test_images, test_labels)
test_loss = mlp.loss(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy*100}%")
print(f"Test Loss: {test_loss:.4f}")

# %%
