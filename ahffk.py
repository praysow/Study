import pandas as pd
import numpy as np
import pickle
import os.path
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from preprocessing import load_bus, load_deay, load_passenger, load_weather
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
import time
# 데이터 로드
bus_csv = load_bus()
passenger_csv = load_passenger()
weather_csv = load_weather()
delay_csv = load_deay()
# 150,151,152,153,154,155,156,157,158,159
# 201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250
# 309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342
# 409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434
# 2511,2512,2513,2514,2515,2516,2517,2518,2519,2520,2521,2522,2523,2524,2525,2526,2527,2528,2529,2530,2531,2532,2533,2534,2535,2536,2537,2538,2539,2540,2541,2542,2543,2544,2545,2546,2547,2548,2549,2550,2551,2552,2553,2554,2555,2556,2557,2558,2559,2560,2561,2562,2563,2564,2565,2566
# 2611,2612,2613,2614,2615,2616,2617,2618,2619,2620,2621,2622,2623,2624,2625,2626,2627,2628,2629,2630,2631,2632,2633,2634,2635,2636,2637,2638,2639,2640,2641,2642,2643,2644,2645,2646,2647,2648,2649
# 2711,2712,2713,2714,2715,2716,2717,2718,2719,2720,2721,2722,2723,2724,2725,2726,2727,2728,2729,2730,2731,2732,2733,2734,2735,2736,2737,2738,2739,2740,2741,2742,2743,2744,2745,2746,2747,2748,2749,2750,2751,2752,2753,2755,2756,2757,2758,2759,2760
# 2811,2812,2813,2814,2815,2816,2817,2818,2819,2820,2821,2822,2823,2824,2825,2826,2827,2828
# df2 = pd.read_csv("c:/_data/시간표.csv")
# df = pd.read_csv("c:/_data/시간표.csv")
# df = df[df['역사코드'].isin([201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250])]
# df.to_csv("c:/_data/2호선.csv")# 5호선 다시
# print(df)

# df = pd.read_csv("c:/_data/2호선.csv")
# df1 = df.drop(['고유번호','주중주말','급행여부','열차코드','열차도착시간'],axis=1)
# df1 = df[df['방향'].str.contains('UP')]
# df1.to_csv("c:/_data/2호선_up.csv")#3호선다시
# print(df1)

# # # df = df.drop(['고유번호','주중주말','급행여부','열차코드','열차도착시간'],axis=1)
# # df = df[df['방향'].str.contains('DOWN')]
# # df.to_csv("c:/_data/2호선_down.csv")
# # print(df)

# 필터링할 숫자 리스트
# numbers = [201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250]

# # 리스트 컴프리헨션을 사용하여 각 숫자를 따옴표로 감싸기
# numbers_filter = [f'{num}' if isinstance(num, int) else f'{num}' for num in numbers]

######
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("c:/_data/3호선_up.csv")

# 필요한 열만 선택
df = df.drop([
            #   '고유번호',
              '호선',
              '역사명',
            #   '주중주말',
              '방향',
            #   '급행여부',
            #   '열차코드',
              '열차도착시간',
              '출발역',
              '도착역'
              ], axis=1)

# # 역사코드별로 데이터프레임 분리
# df1 = df[df['역사코드'].isin([150])].T
# df1 = df1[3:]
# df1 = df1.T
# df2 = df[df['역사코드'].isin([151])].T
# df2 = df2[3:]
# df2 = df2.T
# df3 = df[df['역사코드'].isin([152])].T
# df3 = df3[3:]
# df3 = df3.T
# df4 = df[df['역사코드'].isin([153])].T
# df4 = df4[3:]
# df4 = df4.T
# df5 = df[df['역사코드'].isin([154])].T
# df5 = df5[3:]
# df5 = df5.T
# df6 = df[df['역사코드'].isin([155])].T
# df6 = df6[3:]
# df6 = df6.T
# df7 = df[df['역사코드'].isin([156])].T
# df7 = df7[3:]
# df7 = df7.T
# df8 = df[df['역사코드'].isin([157])].T
# df8 = df8[3:]
# df8 = df8.T
# df9 = df[df['역사코드'].isin([158])].T
# df9 = df9[3:]
# df9 = df9.T
# # # 각 데이터프레임을 열 방향으로 연결하여 하나의 DataFrame으로 합침
# df_c = np.concatenate([df1, df2, df3, df4, df5, df6, df7, df8, df9], axis=1)
# # df1 = df1.drop['역사코드']
# # # CSV 파일로 저장
# # df_c.to_csv("c:/_data/project/1호선_test.csv", index=False)
# print(df_c)
print(df)
# dfs = []
# for i in range(309, 342):
#     df_temp = df[df['역사코드'].isin([i])].T
#     df_temp = df_temp[3:]
#     df_temp = df_temp.T
#     dfs.append(df_temp)

# # 각 데이터프레임의 길이 확인 후 최소 길이 구하기
# min_length = min(len(df_temp) for df_temp in dfs)

# # 최소 길이에 맞춰서 데이터프레임들의 길이를 맞춤
# for i in range(len(dfs)):
#     dfs[i] = dfs[i].iloc[:min_length]

# # numpy 배열로 변환
# arrays = [df_temp.to_numpy() for df_temp in dfs]

# # numpy concatenate를 사용하여 합치기
# df_c = np.concatenate(arrays, axis=1)

# # 결과를 데이터프레임으로 변환
# df_c = pd.DataFrame(df_c)

# # CSV 파일로 저장
# df_c.to_csv("c:/_data/project/up_3호선_test.csv", index=False)
# print(df_c)











# # 데이터 불러오기
# data = pd.read_csv("c:/_data/1호선_down.csv", index_col=0)

# # '고유번호' 열의 데이터 타입 확인
# print(data['역사코드'].dtype)

# # 만약 데이터 타입이 문자열이 아니라면, 문자열로 변환
# if data['역사코드'].dtype != 'object':
#     data['역사코드'] = data['역사코드'].astype(str)

# # 이제 문자열 메서드를 사용하여 필터링할 수 있습니다
# data = data[data['역사코드'].str.contains('158')]
# data = data[data['역사코드'].str.contains('157')]
# data = data[data['역사코드'].str.contains('156')]
# data = data[data['역사코드'].str.contains('155')]
# data = data[data['역사코드'].str.contains('154')]
# data = data[data['역사코드'].str.contains('153')]
# data = data[data['역사코드'].str.contains('152')]
# data = data[data['역사코드'].str.contains('151')]
# data = data[data['역사코드'].str.contains('150')]
# print(data)

# df= data['열차출발시간']
# print(df)
# df.to_csv("c:/_data/dw_1호선_152.csv")

# x1 = pd.read_csv("c:/_data/df_1호선_150.csv",index_col=0)
# x1 = x1.rename(columns={'열차출발시간': '150'})
# x1 = x1[:480]
# x2 = pd.read_csv("c:/_data/df_1호선_151.csv",index_col=0)
# x2 = x2.rename(columns={'열차출발시간': '151'})
# x2 = x2[:480]
# x3 = pd.read_csv("c:/_data/df_1호선_152.csv",index_col=0)
# x3 = x3.rename(columns={'열차출발시간': '152'})
# x3 = x3[:480]
# x4 = pd.read_csv("c:/_data/df_1호선_153.csv",index_col=0)
# x4 = x4.rename(columns={'열차출발시간': '153'})
# x4 = x4[:480]
# x5 = pd.read_csv("c:/_data/df_1호선_154.csv",index_col=0)
# x5 = x5.rename(columns={'열차출발시간': '154'})
# x5 = x5[:480]
# x6 = pd.read_csv("c:/_data/df_1호선_155.csv",index_col=0)
# x6 = x6.rename(columns={'열차출발시간': '155'})
# x6 = x6[:480]
# x7 = pd.read_csv("c:/_data/df_1호선_156.csv",index_col=0)
# x7 = x7.rename(columns={'열차출발시간': '156'})
# x8 = pd.read_csv("c:/_data/df_1호선_157.csv",index_col=0)
# x8 = x8.rename(columns={'열차출발시간': '157'})
# x9 = pd.read_csv("c:/_data/df_1호선_158.csv",index_col=0)
# x9 = x9.rename(columns={'열차출발시간': '158'})
# x = np.concatenate((x1, x2, x3,x4,x5,x6,x7,x8,x9), axis=1)
# print(x1.shape)
# print(x2.shape)
# print(x3.shape)
# print(x4.shape)
# print(x5.shape)
# print(x6.shape)
# print(x7.shape)
# print(x8.shape)
# print(x9.shape)
# x = pd.DataFrame(x)
# x.to_csv("c:/_data/df_1호선_test.csv")

# df = pd.read_csv("c:/_data/평균시간.csv",encoding='EUC-KR')

# import pandas as pd
# import numpy as np

# # 초기 데이터프레임 생성
# x = []

# # 파일 읽기와 전처리
# for i in range(150, 159):  # 파일 번호 150부터 158까지 반복
#     file_path = f"c:/_data/1호선_down_{i}.csv"
#     df = pd.read_csv(file_path, index_col=0)
#     df = df.rename(columns={'열차출발시간': str(i)})  # 열 이름 변경
#     df = df[:480]  # 처음 480개의 행 선택
#     x.append(df)

# # 데이터프레임 병합
# x_merged = pd.concat(x, axis=1)

# # 결과 저장
# x_merged.to_csv("c:/_data/1호선_down_시간.csv")

# print("CSV 파일이 성공적으로 저장되었습니다.")
