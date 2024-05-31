import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__, 'device', DEVICE)

# 1. 데이터
path= "c:\_data\dacon\ddarung\\"
train_csv = pd.read_csv(path+"train.csv",index_col=0)
test_csv= pd.read_csv(path+"test.csv",index_col=0)
submission_csv= pd.read_csv(path+"submission.csv")

# print(train_csv.shape)      (1459, 10)


# train_csv = train_csv.dropna()
train_csv=train_csv.fillna(train_csv.mean())
# train_csv=train_csv.fillna(0)
test_csv=test_csv.fillna(test_csv.mean())
# test_csv=test_csv.fillna(0)


#######x와 y분리#######
x= train_csv.drop(['count'],axis=1)
y= train_csv['count']

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=6)

# StandardScaler는 numpy 배열을 필요로 하므로 변환 전 numpy 배열로 유지
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy 배열을 torch 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_train = torch.FloatTensor(y_train.values).unsqueeze(1).to(DEVICE)  # unsqueeze로 차원 추가
y_test = torch.FloatTensor(y_test.values).unsqueeze(1).to(DEVICE)    # unsqueeze로 차원 추가

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델구성
from torch.utils.data import TensorDataset, DataLoader

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

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
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        return
    #모델구성의 레이어를 만들어주는 곳, 순전파
    def forward(self,input_size):
        x1 = self.linear1(input_size)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        x4 = self.linear4(x3)
        x5 = self.linear5(x4)
        return x5

#model = Model(인풋레이어, 아웃풋레이어) -> 함수형모델과 비슷
model = Model(9,1).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.MSELoss()  # 회귀 문제를 위한 Mean Squared Error Loss
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

epochs = 200
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
rmse = np.sqrt(loss2)
print("Final RMSE:", rmse)

# 예측
with torch.no_grad():
    y_pred = model(x_test).detach().cpu().numpy()

# 성능 평가
r2 = r2_score(y_test.cpu().numpy(), y_pred)

print("R2 score: {:.4f}".format(r2))

'''
Final RMSE: 52.420928797968344
R2 score: 0.5905
'''
