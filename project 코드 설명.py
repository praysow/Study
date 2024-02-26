class CustomImageDataset(Dataset):
    def __init__(self, x_data, y_data, transform=None) -> None:
        """
        클래스의 생성자 메서드입니다.
        
        Parameters:
            x_data (list): 이미지 데이터를 담고 있는 리스트
            y_data (list): 각 이미지의 레이블을 담고 있는 리스트
            transform (callable, optional): 이미지에 적용할 전처리 함수
        """
        self.x_data = x_data  # 이미지 데이터
        self.y_data = y_data  # 레이블 데이터
        self.transform = transform  # 전처리 함수
        
    def __len__(self):
        """
        데이터셋의 길이를 반환하는 메서드입니다.
        """
        return len(self.y_data)  # 레이블 데이터의 길이를 반환합니다.
    
    def __getitem__(self, idx):
        """
        인덱스를 입력받아 해당 인덱스의 이미지와 레이블을 반환하는 메서드입니다.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 텐서를 리스트로 변환합니다.
        
        image = torch.FloatTensor(self.x_data[idx].copy())  # 인덱스에 해당하는 이미지를 가져옵니다.
        label = self.y_data[idx].copy()  # 인덱스에 해당하는 레이블을 가져옵니다.
        sample = image, label  # 이미지와 레이블을 튜플 형태로 묶어 샘플로 반환합니다.
        
        if self.transform:
            sample = self.transform(sample)  # 전처리 함수가 있다면 샘플에 적용합니다.
            
        return sample



class MyLSTM(nn.Module):
    def __init__(self, num_classes, input_shape, hidden_size, num_layers) -> None:
        """
        MyLSTM 클래스의 생성자 메서드입니다.

        Parameters:
            num_classes (int): 출력 클래스의 개수
            input_shape (tuple): 입력 데이터의 모양 (시퀀스 길이, 특성 개수)
            hidden_size (int): LSTM의 은닉 상태 크기
            num_layers (int): LSTM의 층 수
        """
        super(MyLSTM, self).__init__()  # 상위 클래스의 생성자를 호출하여 초기화합니다.

        self.num_classes = num_classes  # 출력 클래스의 개수를 설정합니다.
        self.num_layers = num_layers  # LSTM의 층 수를 설정합니다.
        self.input_size = input_shape[1]  # 입력 데이터의 특성 개수를 설정합니다.
        self.hidden_size = hidden_size  # LSTM의 은닉 상태 크기를 설정합니다.
        self.seq_length = input_shape[0]  # 입력 데이터의 시퀀스 길이를 설정합니다.

        # Conv1d 레이어를 정의합니다.
        self.conv1d = nn.Conv1d(in_channels=self.input_size, out_channels=32, kernel_size=3, stride=1, padding=1)

        # LSTM 레이어를 정의합니다.
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        # Fully connected 레이어를 정의합니다.
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),  # 입력 크기를 128로 변환하는 선형 레이어를 추가합니다.
            nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
            nn.Linear(128, 64),  # 128에서 64로 변환하는 선형 레이어를 추가합니다.
            nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
            nn.BatchNorm1d(64),  # Batch normalization을 수행합니다.
            nn.Dropout(0.01),  # 1%의 드롭아웃을 적용합니다.
            nn.Linear(64, 32),  # 64에서 32로 변환하는 선형 레이어를 추가합니다.
            nn.ReLU(),  # ReLU 활성화 함수를 적용합니다.
            nn.Dropout(0.01),  # 1%의 드롭아웃을 적용합니다.
            nn.Linear(32, num_classes)  # 32에서 num_classes로 변환하는 선형 레이어를 추가합니다.
        )
    
    def forward(self, x):
        """
        순전파 메서드입니다.
        
        Parameters:
            x (torch.Tensor): 입력 데이터

        Returns:
            torch.Tensor: 출력 데이터
        """
        x = self.conv1d(x)  # 입력 데이터에 Conv1d 레이어를 적용합니다.
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # LSTM의 초기 은닉 상태를 설정합니다.
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # LSTM의 초기 셀 상태를 설정합니다.
        ula, (h_out, c_out) = self.lstm(x, (h_0, c_0))  # LSTM 레이어를 적용합니다.
        h_out = h_out.view(-1, self.hidden_size)  # LSTM의 출력을 2D 텐서로 변환합니다.
        out = self.fc(h_out)  # Fully connected 레이어를 적용하여 최종 출력을 계산합니다.
        return out  # 출력을 반환합니다.


if __name__ == '__main__':
    print(f"Using {device} device")  # 현재 사용 중인 디바이스를 출력합니다.
    print(model)  # 모델 구조를 출력합니다.

    EPOCHS = 500  # 에폭 수를 설정합니다.
    best_loss = 987654321  # 초기 최저 손실값을 설정합니다.
    best_model = None  # 초기 최적의 모델을 저장할 변수를 설정합니다.
    patience_count = 0  # 조기 종료를 위한 인내 카운터를 설정합니다.
    for t in range(EPOCHS):  # 설정한 에폭 수만큼 반복합니다.
        print(f"Epoch {t+1}\n---------------------")  # 현재 에폭 번호를 출력합니다.
        train(train_dataloader, model, loss_fn, optimizer, verbose=False)  # 학습을 진행합니다.
        loss = test(test_dataloader, model, loss_fn)  # 테스트 데이터셋에 대한 손실을 계산합니다.
        if loss < best_loss:  # 현재 손실값이 최적의 손실값보다 낮으면
            best_loss = loss  # 최적의 손실값을 업데이트합니다.
            best_model = copy.deepcopy(model)  # 현재 모델을 최적의 모델로 저장합니다.
            patience_count = 0  # 인내 카운터를 초기화합니다.
        else:  # 현재 손실값이 최적의 손실값보다 높으면
            patience_count += 1  # 인내 카운터를 증가시킵니다.
            
        if patience_count >= 100:  # 인내 카운터가 임계치에 도달하면
            print("Early Stopped")  # 조기 종료 메시지를 출력하고
            break  # 반복문을 종료합니다.

    print("===== best model =====")  # 최적의 모델을 출력합니다.
    test(test_dataloader, best_model, loss_fn)  # 최적의 모델을 사용하여 테스트를 수행합니다.
    print("Best loss: ", best_loss)  # 최적의 손실값을 출력합니다.
    print("Done")  # 작업 완료 메시지를 출력합니다.
