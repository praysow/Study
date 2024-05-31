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
path= "c:\\_data\\dacon\\wine\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv = pd.read_csv(path+"test.csv",index_col=0)

# 데이터 인코딩 및 분할
lb = LabelEncoder()
lb.fit(train_csv['type'])
train_csv['type'] = lb.transform(train_csv['type'])
test_csv['type'] = lb.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=8)

# 데이터 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터를 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.LongTensor(y_train.values - 3).to(DEVICE)  # 라벨 범위 조정
y_test = torch.LongTensor(y_test.values - 3).to(DEVICE)    # 라벨 범위 조정

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from torch.utils.data import TensorDataset, DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)

class Model(nn.Module):
    #함수에 들어갈 레이어들의 정의를 넣는곳
    def __init__(self,input_dim,output_dim):
        # super().__init__()가 저장되어있다(아빠)
        super(Model,self).__init__()
        self.linear1 = nn.Linear(input_dim,64)
        self.linear2 = nn.Linear(64,32)
        self.linear3 = nn.Linear(32,16)
        self.linear4 = nn.Linear(16,8)
        self.linear5 = nn.Linear(8,output_dim)
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        return
    #모델구성의 레이어를 만들어주는 곳, 순전파
    def forward(self,input_size):
        x1 = self.linear1(input_size)
        x2 = self.linear2(x1)
        x2 = self.relu(x2)
        x3 = self.linear3(x2)
        x4 = self.linear4(x3)
        x4 = self.relu(x4)
        x5 = self.linear5(x4)
        x6 = self.softmax(x5)
        return x6

#model = Model(인풋레이어, 아웃풋레이어) -> 함수형모델과 비슷
model = Model(12,7).to(DEVICE)

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
    y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

acc = accuracy_score(y_test.cpu().numpy(), y_pred)

print("정확도: {:.4f}".format(acc))

'''
Final loss: 1.6644233465194702
정확도: 0.4855
'''