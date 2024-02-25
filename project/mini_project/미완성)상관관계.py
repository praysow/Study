import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import make_passenger_csv, make_bus_csv, make_delay_csv

def passenger_bus_delay_corr():
    '''
    전철 인원, 버스 인원, 지연시간
    셋 데이터의 상관관계
    '''
    org = make_passenger_csv()
    print(org.tail(3))
    cols = org.columns 
    
    new_passenger = pd.DataFrame()
    

    temp1 = pd.Series(sum(org[col] for col in cols[:10]))
    temp1 = round(temp1/10)

    temp2 = pd.Series(sum(org[col] for col in cols[10:60]))
    temp2 = round(temp1/50)
    
    temp3 = pd.Series(sum(org[col] for col in cols[60:94]))
    temp3 = round(temp1/34)
    
    temp4 = pd.Series(sum(org[col] for col in cols[94:120]))
    temp4 = round(temp1/26)

    temp5 = pd.Series(sum(org[col] for col in cols[120:176]))
    temp5 = round(temp1/56)

    temp6 = pd.Series(sum(org[col] for col in cols[176:215]))
    temp6 = round(temp1/39)

    temp7 = pd.Series(sum(org[col] for col in cols[215:264]))
    temp7 = round(temp1/49)

    temp8 = pd.Series(sum(org[col] for col in cols[264:282]))
    temp8 = round(temp1/18)

    new_passenger['1호선'] = temp1
    new_passenger['2호선'] = temp2
    new_passenger['3호선'] = temp3
    new_passenger['4호선'] = temp4
    new_passenger['5호선'] = temp5
    new_passenger['6호선'] = temp6
    new_passenger['7호선'] = temp7
    new_passenger['8호선'] = temp8


    # print(cols[264:282])
    # print(new_passenger.tail(24))    

    
    # passenger_bus_delay_corr()    



    bus_data = make_bus_csv()
    # 첫 번째와 두 번째 열을 삭제하기
    bus_data = bus_data.iloc[:, 2:]
    delay_data = make_delay_csv()
    
    print(new_passenger.head(3))    
    print('='*100)
    print(bus_data.head(3))    
    print('='*100)
    print(delay_data.head(3))    
    print('='*100)
    
    # 전철, 지연시간 합치기
    passenger_delay_data = pd.concat([new_passenger, delay_data], axis=1)
    
    # 버스, 지연시간 합치기
    bus_delay_data = pd.concat([bus_data, delay_data], axis=1)
    
    # 상관관계 계산
    passenger_matrix = passenger_delay_data.corr()
    bus_matrix = bus_delay_data.corr()
    
       
    # 상관관계 행렬 출력
    print("전철 상관관계 상위 5 행 :\n", passenger_matrix.head(5))    
    print("버스 상관관계 상위 5 행 :\n", bus_matrix.head(5))    
    
    return passenger_matrix, bus_matrix



def plot_correlation_heatmap(passenger_matrix, bus_matrix):
    from matplotlib import font_manager, rc
    
    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    
    plt.figure(figsize=(12, 6))

    # 전철 상관관계 히트맵
    plt.subplot(1, 2, 1)
    sns.heatmap(passenger_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap - Passenger & Delay')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    # 버스 상관관계 히트맵
    plt.subplot(1, 2, 2)
    sns.heatmap(bus_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap - Bus & Delay')
    plt.xlabel('Variables')
    plt.ylabel('Variables')

    plt.tight_layout()
    plt.show()

# 데이터 준비 및 상관관계 계산
passenger_matrix, bus_matrix = passenger_bus_delay_corr()


# 시각화
plot_correlation_heatmap(passenger_matrix, bus_matrix)
