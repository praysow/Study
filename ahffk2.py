import pandas as pd

# Read the CSV file
df = pd.read_csv("c:/_data/project/up_7호선_평균.csv", encoding='EUC-KR')

# '평균시간' 열을 timedelta 형식으로 변환
df['평균시간'] = pd.to_timedelta(df['평균시간'])

# '평균시간'이 '20:00:00' 이상인 행 제거
df = df[df['평균시간'] < pd.to_timedelta('20:00:00')]

# 시간대별로 평균 계산
result_data = []
for i in range(5, 24):  # 시간대 05:00:00 ~ 12:00:00까지 반복
    start_time = f'{i}:00:00'  # 시작 시간 설정
    end_time = f'{i}:59:59'  # 종료 시간 설정
    
    # 해당 시간대의 데이터 필터링
    filtered_data = df[df['1'].between(start_time, end_time)]
    
    # 필터링된 행이 없을 경우 처리
    if not filtered_data.empty:
        # 필터링된 행 중에서 최대 6개 행 선택
        filtered_data = filtered_data.head(6)
        
        # 평균 계산
        average_time = filtered_data['평균시간'].mean()
        
        # 결과를 리스트에 추가
        result_data.append({'시작 시간': start_time, '평균 시간': average_time})
        print(f'{start_time} 평균:', average_time)  # 결과 출력
    else:
        result_data.append({'시작 시간': start_time, '평균 시간': '데이터 없음'})
        print(f'{start_time}사이의 데이터가 없습니다.')  # 결과 출력

# 결과 데이터프레임 생성
result_df = pd.DataFrame(result_data)

# 결과를 CSV 파일로 저장
result_df.to_csv("c:/_data/4호선_평균시간_결과.csv", index=False)

print("CSV 파일이 성공적으로 저장되었습니다.")