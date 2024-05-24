import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print(torch.__version__,'device',DEVICE)

#1.데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)

print(x.shape,y.shape)
# torch.Size([3, 1]) torch.Size([3])
#2.모델구성
# model = Sequential()
# model.add(Dense(1,input_dim=1))
model = nn.Linear(1,1).to(DEVICE) #input,output    y = xw + b


#3.컴파일,훈련
# model.compile(loss='mse',optimizer='adam')
criterion = nn.MSELoss()    #criterion:표준
optimizer = optim.Adam(model.parameters(),lr=0.1)
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

epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model,criterion,optimizer,x,y)
    print('epochL{},loss{}'.format(epoch,loss))
    
#평가 예측
# loss = model.evaluate(x,y)
def evaluate(model,criterion,x,y):
    model.eval()    #평가모드
    with torch.no_grad():
        y_pred = model(x)
        loss2 = criterion(y, y_pred)
        return loss2.item()
    
loss2 = evaluate(model,criterion,x,y)
print("duck loss",loss2)

#result = model.predict([4])
result = model(torch.Tensor([[4]]).to(DEVICE))
print('4의 예측값',result.item())
