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
path= "c:\_data\dacon\dechul\\"
train_csv = pd.read_csv(path + "train.csv", index_col=0)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
sample_csv = pd.read_csv(path + "sample_submission.csv")

x = train_csv.drop(['대출등급'], axis=1)
y = train_csv['대출등급']

y = pd.get_dummies(y)
# LabelEncoder를 사용하여 문자열 라벨을 숫자로 인코딩
lb = LabelEncoder()
all_data = pd.concat([x, test_csv])  # 학습 데이터와 테스트 데이터를 합침
for column in x.columns:
    if all_data[column].dtype == 'object':
        all_data[column] = lb.fit_transform(all_data[column])

# 다시 학습 데이터와 테스트 데이터로 나눔
x_encoded = all_data.iloc[:len(x)]
test_csv_encoded = all_data.iloc[len(x):]

x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, train_size=0.9, random_state=333, stratify=y)

# 데이터 표준화
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 데이터를 텐서로 변환
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

# y_train을 정수로 변환하여 텐서로 변환
y_train = torch.FloatTensor(y_train.values).to(DEVICE)
y_test = torch.FloatTensor(y_test.values).to(DEVICE)


# 2. 모델 구성

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
model = Model(13,7).to(DEVICE)

# 3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = criterion(hypothesis, y)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 2000
for epoch in range(1, epochs + 1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 10 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss))

# 평가 및 예측
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        loss = criterion(y_pred, y)
        return loss.item()

loss2 = evaluate(model, criterion, x_test, y_test)
print("최종 Loss:", loss2)

# 예측 및 성능 평가
with torch.no_grad():
    y_pred = model(x_test)
    y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy()

y_test = torch.argmax(y_test, dim=1).cpu().numpy()

acc = accuracy_score(y_test, y_pred)

print("정확도: {:.4f}".format(acc))

'''
최종 Loss: 1.5465037822723389
정확도: 0.6189
'''