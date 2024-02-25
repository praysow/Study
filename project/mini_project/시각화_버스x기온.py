'''
실제 import되어 사용되는 코드가 아니라 데이터를 분석하기 위해 만든 파일입니다

시각적으로 화려하게 하라고 하셨으니 plt로 분석자료를 많이 띄워야합니다
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
            


def busXweather_graph():
    '''
    1월~8월 일별 버스 이용인원 파일 분석 함수
    '''
    # 원하는 컬럼들을 리스트로 지정합니다.
    desired_columns = ["사용일자", "승차총승객수", "하차총승객수",]
                       

    # 원하는 컬럼만 선택하여 파일을 읽어옵니다.
    user_csv = pd.read_csv("./data/2023년1~8월 일별 버스 이용객수2.csv",
                                usecols=desired_columns, 
                                parse_dates=['사용일자'],
                                # encoding="euc-kr",
                                # index_col=0
                                )
    
    
    '''승차 하차 데이터 선택'''
    user_on_data = user_csv['승차총승객수']  # 
    user_off_data = user_csv['하차총승객수']  # 
    

    '''
    날씨 파일 분석 함수
    '''
    # 원하는 컬럼들을 리스트로 지정합니다.
    desired_columns = ["일시", "기온(°C)", "강수량(mm)", "적설(cm)"]

    # 원하는 컬럼만 선택하여 파일을 읽어옵니다.
    weather_csv = pd.read_csv("./data/SURFACE_ASOS_108_HR_2023_2023_2024.csv",
                                usecols=desired_columns, 
                                parse_dates=['일시'],
                                encoding="euc-kr")
    
    # 2023-08-31까지의 데이터만 필터링합니다.
    weather_csv_filtered = weather_csv[weather_csv['일시'] <= '2023-08-31']
    
    
    '''온도 데이터 선택'''
    temperature_data = weather_csv_filtered['기온(°C)']  # 기온 데이터만 선택
    
    #######################################################################
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc
    
    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    # 박스 플롯 대신 막대 그래프를 사용하여 버스 이용객 수를 시각화합니다.
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.bar(user_csv['사용일자'], user_csv['승차총승객수'], label='승차총승객수', color='skyblue', alpha=0.7)
    ax1.bar(user_csv['사용일자'], user_csv['하차총승객수'], label='하차총승객수', color='orange', alpha=0.7)
    ax1.set_xlabel('날짜')
    ax1.set_ylabel('이용객 수                 \n(단위 백만)                 ', rotation = 0)
    ax1.set_title('일별 버스 이용객 수')
    ax1.tick_params(axis='x', rotation=45)  # x축 라벨 회전
    ax1.legend()

    # 기온 데이터를 같은 그림에 추가합니다.
    ax2 = ax1.twinx()
    ax2.plot(weather_csv_filtered['일시'], temperature_data, color='red', marker='o', linestyle='-', label='기온(°C)', markersize = 1)
    ax2.set_ylabel('기온 (°C)', rotation = 0)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()    

busXweather_graph()


# if __name__ == '__main__':
#     # congestion_file_analyze()
#     user_file_analyze()
# user_file_analyze()
