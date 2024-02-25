import numpy as np
import pandas as pd

def day_rain_mean():
    '''
    강수량 일별 평균 분석 함수
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
    
    # 일별 강수량 평균을 계산합니다.
    daily_rain_avg = weather_csv_filtered.groupby(weather_csv_filtered['일시'].dt.date)['강수량(mm)'].mean()
    
    return daily_rain_avg
    

def day_trail_mean():
    '''
    지하철 이용객 일별 평균 분석 함수
    '''
    trail_csv = pd.read_csv('./data/2023년 1~8월 이용인원.csv',
                            parse_dates=['날짜'],
                            )    
    
    # 2023-08-31까지의 데이터만 필터링합니다.
    trail_csv_filtered = trail_csv[trail_csv['날짜'] <= '2023-08-31']
    daily_trail_avg = round(trail_csv_filtered.groupby(trail_csv_filtered['날짜'].dt.date)['합 계'].mean())
    daily_trail_avg = daily_trail_avg.astype(int)
    
    return daily_trail_avg
    

##############################################################################
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 경로 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 사용하고자 하는 한글 폰트 경로로 변경해주세요
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 마이너스 부호 깨짐 방지 설정
plt.rcParams['axes.unicode_minus'] = False


# 위에서 제공한 두 함수를 사용하여 데이터를 가져옵니다.
daily_rain_avg = day_rain_mean()
daily_trail_avg = day_trail_mean()

# 비가 오는 날과 오지 않는 날의 평균 이용객 수를 계산합니다.
rainy_days_trail_avg = (daily_trail_avg[daily_rain_avg > 0].mean())/2000
non_rainy_days_trail_avg = (daily_trail_avg[daily_rain_avg == 0].mean())/2000

# 시각화를 위한 데이터 준비
labels = ['우천시', '평상시']
trail_avgs = [rainy_days_trail_avg, non_rainy_days_trail_avg]

# 막대 그래프로 시각화

# 차이를 표시할 텍스트 설정
diff_text = f'차이: {abs(rainy_days_trail_avg - non_rainy_days_trail_avg):.2f}'


plt.figure(figsize=(10, 6))

bars = plt.bar(labels, trail_avgs, color=['lightblue', 'orange'], width=0.2)

plt.bar(labels, trail_avgs, color=['lightblue', 'orange'], width=0.2)
# 첫 번째 막대에 텍스트 추가
plt.text(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height(), diff_text,
         ha='center', va='bottom', color='black')

plt.title('우천시와 평상시 지하철 평균 이용객수')
plt.ylabel('지하철 평균 이용객수            \n(단위 백만)                ', rotation=0)
plt.show()





























# # 그래프 크기 설정
# plt.figure(figsize=(12, 6))

# # 강수량 데이터를 실선 그래프로 플로팅합니다.
# plt.plot(rain_day_avg.index, rain_day_avg.values, color='blue', linestyle='-', marker='o', label='강수량(mm)')

# # x 축 레이블 설정
# plt.xlabel('날짜')

# # y 축 레이블 설정 (왼쪽 y 축)
# plt.ylabel('지하철 평균 이용객 수                             ', rotation=0)

# # 지하철 평균 이용객 수를 막대 그래프로 플로팅합니다.
# plt.bar(trail_day_avg.index, trail_day_avg.values, color='orange', alpha=0.5, label='지하철 평균 이용객 수')

# # 오른쪽 y 축에 강수량 표시
# plt.twinx()
# plt.ylabel('                     강수량(mm)', rotation=0)

# # 그래프 제목 설정
# plt.title('일별 강수량 및 지하철 평균 이용객 수')

# # 범례 표시
# plt.legend()

# # x 축 눈금 라벨 회전
# plt.xticks(rotation=45)

# # 그리드 표시
# # plt.grid(True, linestyle='--', alpha=0.7)

# # 그래프 출력
# plt.tight_layout()
# plt.show()
