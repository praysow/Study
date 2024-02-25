import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from keras.datasets import fashion_mnist
import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from preprocessing import load_bus, load_delay, load_passenger, load_weather
from function_package import split_xy
from sklearn.metrics import r2_score
from torcheval.metrics import R2Score
import copy

print(torch.__version__)    # 2.2.0+cu118

# x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]]).reshape(-1,3,1)
# y = np.array([4,5,6,7,8,9,10]).astype(np.float32)
# print(x.shape,y.shape) # (7, 3, 1) (7,)

# bus_csv = load_bus()
# print(bus_csv.shape)
# x, y = split_xy(bus_csv,100)
# print(x.shape, y.shape)
# print(y[:10])

# delay_csv = load_delay()
# x, y = split_xy(delay_csv,100)

passenger_csv = load_passenger()
x, y = split_xy(passenger_csv,24)
print(x.shape,y.shape)

np.save('./data/temp_x.npz',x)
np.save('./data/temp_y.npz',y)

x = np.load('./data/temp_x.npz')
y = np.load('./data/temp_y.npz')

class CustomImageDataset(Dataset):
    def __init__(self,x_data,y_data,transform=None) -> None:    # 생성자, x,y데이터, 변환함수 설정
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        
    def __len__(self):          # 어떤 값을 길이로서 반환할지
        return len(self.y_data)
    
    def __getitem__(self, idx): # 인덱스가 들어왔을 때 어떻게 값을 반환할지 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = torch.FloatTensor(self.x_data[idx].copy())  # torchTensor 형식으로 변환
        label = self.y_data[idx].copy()                     # 얘는 9를 변환하면 사이즈9짜리 텐서로 만들어버리기에 그냥 이대로 사용
        sample = image, label                               # 반드시 x, y 순으로 반환할 것
        
        if self.transform:  # 전처리 함수 적용
            sample = self.transform(sample)
            
        return sample

import matplotlib.pyplot as plt

training_data = CustomImageDataset(x,y) # 인스턴스 선언 및 데이터 지정

BATCH_SIZE = 128
train_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE, pin_memory=True)  # 만든 커스텀 데이터를 iterator형식으로 변경
test_dataloader = DataLoader(training_data,batch_size=BATCH_SIZE, pin_memory=True)

for X, y in test_dataloader:    # dataloader는 인덱스로 접근이 되지 않으며 .next()또한 사용할 수 없다 오직 for문만 가능하며 그렇기에 출력을 위해 한바퀴 돌자마자 break한다
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shaep of y: {y.shape} {y.dtype}")
    break


device = (
    "cpu"
    # "cuda"
    # if torch.cuda.is_available() 
    # else "mps" 
    # if torch.backends.mps.is_available() 
    # else "cpu"
)

class TorchLSTM(nn.Module):
    def __init__(self,input_shape,output_shape) -> None:
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.LSTM(input_shape,256,batch_first=True),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,output_shape)
        )
        
    def forward(self,x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class MyLSTM(nn.Module):
    def __init__(self, num_classes, input_shape, hidden_size, num_layers) -> None:
        super(MyLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers  = num_layers
        self.input_size  = input_shape[1]
        self.hidden_size = hidden_size
        self.seq_length  = input_shape[0]
        # self.lstm = nn.LSTM(input_size=input_shape[1], hidden_size=hidden_size,
        #                     num_layers=num_layers, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.01),
            nn.Linear(64,32),
            nn.ReLU(),
            # nn.Linear(32,16),
            # nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(32,num_classes)
        )
    
    def forward(self, x):
        x = self.conv1d(x)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        ula, (h_out, c_out) = self.lstm(x, (h_0,c_0))
        h_out = h_out.view(-1, self.hidden_size)   
        out = self.fc(h_out)
        return out
    
# model = TorchLSTM(1,(3,1),1).to(device)
model = MyLSTM(num_classes=1, input_shape=(x.size(1),x.size(2)), hidden_size=128, num_layers=1).to(device)

loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss() # 이 함수에 softmax가 내재되어있기에 모델에서 softmax를 쓰면 안된다
optimizer = torch.optim.Adam(model.parameters())

def train(dataloader, model, loss_fn, optimizer, verbose=True):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 예측 오류 계산
        pred = model(X)
        pred = pred.to(device)
        loss = loss_fn(pred, y)
        
        # 역전파
        optimizer.zero_grad()   # 역전파하기 전에 기울기를 0으로 만들지 않으면 전의 학습이 영향을 준다
        loss.backward()         # 역전파 계산
        optimizer.step()        # 역전파 계산에 따라 파라미터 업데이트(이 시점에서 가중치가 업데이트 된다)
        
        if (batch % 10 == 0) and verbose:
            loss, current = loss.item(), (batch+1)*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # print("size check: ",size,num_batches) # size는 테스트 데이터의 개수, num_batches는 batch_size에 의해 몇 바퀴 도는지
    model.eval()    # 모델을 평가 모드로 전환, Dropout이나 BatchNomalization등을 비활성화
    test_loss, correct = 0, 0
    
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.reshape(-1)
            pred = pred.to(device)
            test_loss += loss_fn(pred, y).item()
            metric = R2Score()
            metric.update(pred,y).to(device)
            r2 = metric.compute()
            
            correct += r2 
    test_loss /= num_batches    
    correct /= num_batches
    print(f"Test Error \n R2: {(correct):>0.4f}, Avg loss: {test_loss:>8f}\n")
    return test_loss
    
if __name__ == '__main__':
    print(f"Using {device} device")
    print(model)
    
    EPOCHS = 500
    best_loss = 987654321
    best_model = None
    patience_count = 0
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n---------------------")
        train(train_dataloader, model, loss_fn, optimizer,verbose=False)
        loss = test(test_dataloader, model, loss_fn)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model)   # restore best weight 구현
            patience_count = 0
        else:
            patience_count += 1
            
        if patience_count >= 100:
            print("Early Stopped")
            break

    print("===== best model =====")
    test(test_dataloader,best_model,loss_fn)
    print("Best loss: ",best_loss)
    print("Done")
    
    # print(x[:10],y[:10])
    # with torch.no_grad():
    #     model.eval()
    #     pred = model(x)
    #     metric = R2Score()
    #     metric.update(x,y)
    #     metric.compute()
    # import os
    # dir_path = os.getcwd()
    # print(dir_path)
    # torch.save(model.state_dict(), dir_path+"./python/torch_model_save/torch_LSTM_model.pth")
    # print("Model saved")