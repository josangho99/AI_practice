import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.utils.data import random_split

print("PyTorch version:[%s]." % (torch.__version__))
print("torchvision version:[%s]." % (torchvision.__version__))
print("GPU 사용 가능 여부: {}".format(torch.cuda.is_available()))
print("CUDA 버전: {}".format(torch.version.cuda))


USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# for reproducibility
random.seed(777)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 15
batch_size = 100
learning_rate = 0.1
input_size = 784    # 28x28 image       # 고정된 값 (이미지 크기)
hidden_size = 100   # 임의의 값           # 임의의 값 (hidden layer의 노드 개수)
output_size = 10    # 0~9               # 고정된 값 (분류할 클래스 개수)

mnist_train = dsets.MNIST(
    root="MNIST_data/",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)

mnist_test = dsets.MNIST(
    root="MNIST_data/",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
)
train_size = 55000
val_size = 5000
train, val = random_split(mnist_train, [train_size, val_size], generator = torch.Generator.manual_seed(777))

# dataset loader
data_loader = DataLoader(
    dataset = train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,  # 배치 크기는 100
)

val_loader = DataLoader(
    dataset = val,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,  # 배치 크기는 100
)

# Print sample data
# 첫번째 미니배치의 첫번째 이미지 데이터를 불러옴
# shape: [배치크기, 채널, 높이, 너비]
# 채널: 1은 흑백, 3은 RGB
for X, Y in data_loader:
    print('X:', X.size(), 'type:', X.type())
    print('Y:', Y.size(), 'type:', Y.type())

    plt.imshow(X[0, 0, :, :].numpy(), cmap='gray')
    plt.title('Class: ' + str(Y[0].item()))
    plt.show()

    break


class MLP(nn.Module):
    """
    Model : 3-layered Neural Network (Multi-layer Perceptron)

    Linear -> Sigmoid -> Linear -> Sigmoid -> Linear -> Softmax

    ** 참고: pytorch에서는 softmax를 loss function에서 포함하고 있음 (따로 정의하지 않아도 됨)
    """

    def __init__(self):
        super(MLP, self).__init__()

        # 여기에 MLP 모델을 정의해주세요
        self.sequential = nn.Sequential(

            nn.Flatten(start_dim=1, end_dim=-1),   # 28x28 image를 784x1 vector로 변환
            nn.Linear(input_size, hidden_size,bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size,bias=True),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size,bias=True)
        )

    def forward(self, x):
        x = self.sequential(x)

        return x  # 마지막 layer의 output을 리턴
    
# 작성한 모델 정의
model = MLP().to(device)
print(model)

# 비용 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부적으로 소프트맥스 함수를 포함하고 있음.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(training_epochs):  # 앞서 training_epochs의 값은 15로 지정함.
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:

        # 배치 크기가 100이므로 아래의 연산에서 X는 (100, 784)의 텐서가 된다.
        # X = X.view(-1, 28 * 28).to(device)    # model.forward()에서 nn.Flatten()을 사용하므로 필요 없음.
        X = X.to(device)

        # 레이블은 원-핫 인코딩이 된 상태가 아니라 0 ~ 9의 정수.
        Y = Y.to(device)    # 원-핫 인코딩을 하지 않아도 nn.CrossEntropyLoss()를 사용할 수 있음. (torch에서 자동으로 해줌.)

        optimizer.zero_grad()

        hypothesis = model(X)            # hypothesis : y_hat (예측값)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
        if(epoch % 5 == 0):
            val_batch = len(val_loader)
            for X, Y in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                optimizer.zero_grad()
                hypothesis = model(X)
                cost = criterion(hypothesis, Y)
                avg_cost += cost / total_batch
                print("cost =", "{:.9f}".format(avg_cost))

    
print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
print("Learning finished")

# 테스트 데이터를 사용하여 모델을 테스트한다.
with torch.no_grad():  # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    # X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    X_test = mnist_test.test_data.float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy:", accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    # X_single_data = mnist_test.test_data[r : r + 1].view(-1, 28 * 28).float().to(device)
    X_single_data = mnist_test.test_data[r : r + 1].float().to(device)
    Y_single_data = mnist_test.test_labels[r : r + 1].to(device)

    print("Label: ", Y_single_data.item())
    single_prediction = model(X_single_data)
    print("Prediction: ", torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r : r + 1].view(28, 28), cmap="Greys", interpolation="nearest")
    plt.show()