import numpy as np
import pandas as pd

def weather_file_analyze():
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
    
    
    '''읽어온 데이터 정보를 출력합니다.'''
    print(weather_csv_filtered.info())
    #  0   일시       5809 non-null   datetime64[ns]
    #  1   기온(°C)   5809 non-null   float64
    #  2   강수량(mm)  809 non-null    float64
    #  3   적설(cm)   76 non-null     float64
    
    
    '''데이터를 출력.'''
    print(weather_csv_filtered.head(3))
    #                  일시  기온(°C)  강수량(mm)  적설(cm)
    # 0 2023-01-01 00:00:00     0.9      NaN     NaN
    # 1 2023-01-01 01:00:00     1.5      NaN     NaN
    # 2 2023-01-01 02:00:00     1.5      NaN     NaN

    
    def outliers(data_out):
        quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])  # 퍼센트 지점
        print('1사분위 : ', quartile_1)
        print('q2 : ', q2)
        print('3사분위 : ', quartile_3)
        iqr = quartile_3 - quartile_1   # 이상치 찾는 인스턴스 정의
        # 최대값이 이상치라면 최대값최소값으로 구하는 이상치는 이상치를 구한다고 할수없다
        print('iqr : ', iqr)
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        # -10 의 1.5 범위만큼 과 50의 1.5 범위만큼을 이상치로 생각을 하고 배제
        # 4~10 까지는 안전빵이라고 정의

        # 조건문(인덱스 반환) 
        return np.where((data_out > upper_bound) | (data_out < lower_bound))


    # 기온 데이터에 대한 이상치를 찾습니다.
    outliers_indices = outliers(temperature_data)
    print("이상치 인덱스:", outliers_indices)

    
    # 박스 플롯 그리기
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, rc

    # 한글 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    
    # 마이너스 부호 깨짐 방지 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    # 그림의 크기 설정
    plt.figure(figsize=(8, 8))
    plt.boxplot(temperature_data)
    plt.ylabel('기온 (°C)        ', rotation=0)
    # plt.xlabel('123')
    plt.title('기온 데이터의 박스플롯')
    plt.show()


# 함수 호출
weather_file_analyze()

