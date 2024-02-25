'''
실제 import되어 사용되는 코드가 아니라 데이터를 분석하기 위해 만든 파일입니다

시각적으로 화려하게 하라고 하셨으니 plt로 분석자료를 많이 띄워야합니다
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# def congestion_file_analyze():
#     '''
#     혼잡도 파일 분석 함수입니다
#     congestion : 혼잡도
#     '''
#     congestion_csv = pd.read_csv("./data/서울교통공사_지하철혼잡도정보_20221231.csv",index_col=0,encoding="EUC-KR")
#     print(congestion_csv.info)
    
#     ''' 각 호선별 최대 혼잡도 계산 '''
#     max_congestion = {}
#     for i in range(congestion_csv.shape[0]): # 호선별 최대치 생성
#         data = congestion_csv.iloc[i]
#         name = int(data['호선'])
#         real_data = data.loc['5시30분':'00시30분'].copy()
#         if name in max_congestion:
#             max_congestion[name] = max(max_congestion[name], real_data.max()) # 기존데이터와 신규 데이터중 큰쪽으로 대체
#         else:
#             max_congestion[name] = real_data.max()

#     print("각 호선별 최대 혼잡도",max_congestion)
#     # 각 호선별 최대 혼잡도 {1: 107.8, 2: 172.3, 3: 154.8, 4: 185.5, 5: 140.9, 6: 113.8, 7: 160.6, 8: 136.8}
    
#     ''' 각 호선 별 최대 승객수 계산'''
#     max_transfer = {}
#     for key, value in max_congestion.items():
#         if key <= 4:
#             max_transfer[key] = round(160 * 10 * value)
#         elif key == 8:
#             max_transfer[key] = round(160 * 6 * value)
#         else:
#             max_transfer[key] = round(160 * 8 * value)
#     print("각 호선별 최대 승객수",max_transfer)
#     # 각 호선 별 최대 승객수 {1: 172480, 2: 275680, 3: 247680, 4: 296800, 5: 180352, 6: 145664, 7: 205568, 8: 131328}
            


def bus_user_file_analyze():
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
    
    
    '''읽어온 데이터 정보를 출력합니다.'''
    print(user_csv.info())
    #  #   Column  Non-Null Count  Dtype
    # ---  ------  --------------  -----
    #  0   사용일자    242 non-null    datetime64[ns]
    #  1   승차총승객수  242 non-null    int64
    #  2   하차총승객수  242 non-null    int64    
    
    
    '''데이터를 출력.'''
    print(user_csv.head(3))
    #   사용일자   승차총승객수   하차총승객수
    # 0 2023-01-01  2310197  2257688
    # 1 2023-01-02  4413450  4319065
    # 2 2023-01-03  4625953  4527318    
    

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


    # 승차, 하차 데이터에 대한 이상치를 찾습니다.
    outliers_indices = outliers([user_on_data, user_off_data])
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
    plt.figure(figsize=(10, 10))
    plt.boxplot([user_on_data,user_off_data], labels=['승차총승객수', '하차총승객수'])
    plt.ylabel('승객수                  \n(단위 백만)                  ', rotation=0)
    # plt.xlabel('123')
    plt.title('일별 버스 이용객 승하차수의 박스플롯')
    plt.show()


# 함수 호출
bus_user_file_analyze()





# if __name__ == '__main__':
#     # congestion_file_analyze()
#     user_file_analyze()
# user_file_analyze()
