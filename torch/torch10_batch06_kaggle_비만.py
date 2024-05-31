import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

# CUDA 사용 가능 여부 확인
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# 1. 데이터 로드 및 전처리
path = "c:/_data/kaggle/비만/"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sample = pd.read_csv(path + "sample_submission.csv")
x = train.drop(['NObeyesdad'], axis=1)
y = train['NObeyesdad']

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
lb = LabelEncoder()
for column in columns_to_encode:
    x[column] = lb.fit_transform(x[column])
    test[column] = lb.fit_transform(test[column])

# 라벨을 정수로 변환
y = lb.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=367, stratify=y, shuffle=True)

# 데이터 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터를 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# y_train을 정수로 변환하여 텐서로 변환
y_train = torch.LongTensor(y_train).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

from torch.utils.data import TensorDataset, DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=300, shuffle=True)
test_loader = DataLoader(test_set, batch_size=300, shuffle=False)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 8)
        self.linear5 = nn.Linear(8, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_size):
        x = self.relu(self.linear1(input_size))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.linear5(x)
        x = self.softmax(x)
        return x

model = Model(16, 7).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    model.train()  # 훈련 모드
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()  # 0으로 초기화
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()  # 기울기 계산
        optimizer.step()  # 가중치 수정
        total_loss += loss.item()
        
    return total_loss / len(loader)

epochs = 1000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 10 == 0:  # 10 에포크마다 로그 출력
        print('epoch: {}, loss: {}'.format(epoch, loss))

# 평가 예측
def evaluate(model, criterion, loader):
    model.eval()  # 평가 모드
    total_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            
    return total_loss / len(loader)

loss2 = evaluate(model, criterion, test_loader)
print("Final loss:", loss2)

# 예측 및 성능 평가
with torch.no_grad():
    y_pred = model(x_test)
    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

# y_test를 호스트 메모리로 복사
y_test = y_test.cpu().numpy()

# 정확도 계산
acc = accuracy_score(y_test, y_pred)
print("정확도: {:.4f}".format(acc))

'''
Final loss: 1.5745738915034704
정확도: 0.5906
'''
