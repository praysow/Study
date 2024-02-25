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
    
    # 데이터프레임 합치기
    combined_data = pd.concat([passenger_data, bus_data, delay_data], axis=1)
    
    # 상관관계 계산
    corr_matrix = combined_data.corr()
    
    # 상관관계 행렬 출력
    print("상관관계 상위 5 행 :\n", corr_matrix.head(5))    
    print("상관관계 하위 5 행 :\n", corr_matrix.tail(5))    
    
    
passenger_bus_delay_corr()