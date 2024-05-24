import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

print(x,y)

#2.모델구성
# model = Sequential()
# model.add(Dense(1,input_dim=1))
model = nn.Linear(3,1) #input,output    y = xw + b


#3.컴파일,훈련
# model.compile(loss='mse',optimizer='adam')
criterion = nn.MSELoss()    #criterion:표준
optimizer = optim.Adam(model.parameters(),lr=0.01)
#model.fit(x,y,epochs=100,batch_size=1)

#평가모드에서는 batch normal,drop out 사용x
#그래서 model.eval을 사용(predict할때)
def train(model,criterion, optimizer, x, y):
    # model.train()   #훈련모드, 디폴트
    optimizer.zero_grad()       #0으로 초기화시킴
    hypothesis = model(x)
    loss = criterion(hypothesis,y)
    #여기까지 순전파
    loss.backward() #loss를 weight로 미분하겠다, 기울기(gradient)값 계산까지
    optimizer.step() #가중치(w) 수정
    #여기까지 역전파
    return loss.item()  #loss만 사용하면 torch데이터로 나온다(item을 사용해야 numpy데이터로 나옴)

epochs = 200
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print('epochL{},loss{}'.format(epoch,loss))
    
#평가 예측
# loss = model.evaluate(x,y)
def evaluate(model,criterion,optimizer,x,y):
    model.eval()    #평가모드
    