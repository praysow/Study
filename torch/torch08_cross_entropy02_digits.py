import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

print(np.unique(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8)

# StandardScaler는 numpy 배열을 필요로 하므로 변환 전 numpy 배열로 유지
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy 배열을 torch 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)  # LongTensor로 변환 longtensor는 정수로 변환해주기때문에 onehotencoding을 할 필요가 없다
y_test = torch.LongTensor(y_test).to(DEVICE)    # LongTensor로 변환

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델구성
model = nn.Sequential(
    nn.Linear(64, 32),  # input 크기를 64로 설정 (피처 개수)
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)   # 최종 출력 노드를 10으로 설정 (클래스 개수)
).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()  # 훈련 모드
    optimizer.zero_grad()  # 0으로 초기화
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()  # 기울기 계산
    optimizer.step()  # 가중치 수정
    return loss.item()  # loss를 numpy 데이터로 반환

epochs = 200
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 10 == 0:  # 10 에포크마다 로그 출력
        print('epoch: {}, loss: {}'.format(epoch, loss))

# 평가 예측
def evaluate(model, criterion, x, y):
    model.eval()  # 평가 모드
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        return loss.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("Final Loss:", loss2)

# 예측
with torch.no_grad():
    y_pred = model(x_test)
    y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

# 성능 평가
acc = accuracy_score(y_test.cpu().numpy(), y_pred)

print("Accuracy score: {:.4f}".format(acc))

'''
Final Loss: 0.1931007206439972
Accuracy score: 0.9667
'''
