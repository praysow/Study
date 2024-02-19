import pandas as pd
from datetime import datetime       #시계열에서 날짜종류를 사용할때 사용
import numpy as np

'''
결측치 처리
1.행 또는 열 삭제
2.임의의 값
 평균: mean
 중위값 : median
 0 : fillna
 앞값 : ffill
 뒷값 : bfill
 특정값 : 777(먼가 조건을 같이 넣는게 좋다)
 기타등등
3.보간 : interpolate
4.모델 : predict
5.부스팅 계열 : 통상 결측치 이상치에 대해 자유롭다
'''


dates = ['2/16/2024','2/17/2024','2/18/2024','2/19/2024','2/20/2024','2/21/2024']
dates = pd.to_datetime(dates)
print(dates)
print('===========================')
ts = pd.Series([2,np.nan,np.nan,8,10,np.nan],index =dates)
print(ts)

ts = ts.interpolate()
print(ts)