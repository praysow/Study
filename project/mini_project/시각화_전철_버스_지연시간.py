import numpy as np
import pandas as pd
from preprocessing import make_passenger_csv, make_bus_csv, make_delay_csv
def passenger_bus_delay_corr():
    '''
    전철 인원, 버스 인원, 지연시간
    셋 데이터의 상관관계
    '''
    passenger_data = make_passenger_csv()
    bus_data = make_bus_csv()
    delay_data = make_delay_csv()
    
    print(passenger_data.head(3))    
    print('='*100)
    print(bus_data.head(3))    
    print('='*100)
    print(delay_data.head(3))    
    print('='*100)
    
    # 전철, 지연시간 합치기
    passenger_delay_data = pd.concat([passenger_data, delay_data], axis=1)
    
    # 버스, 지연시간 합치기
    bus_delay_data = pd.concat([bus_data, delay_data], axis=1)
    
    # 상관관계 계산
    passenger_matrix = passenger_delay_data.corr()
    bus_matrix = bus_delay_data.corr()
    
    # 상관관계 행렬 출력
    print("전철 상관관계 상위 5 행 :\n", passenger_matrix.head(5))    
    print("버스 상관관계 상위 5 행 :\n", bus_matrix.head(5))    
    
    
    
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(passenger_matrix, bus_matrix):
    # 전철 상관관계 히트맵
    plt.figure(figsize=(12, 6))
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

    
    
passenger_bus_delay_corr()