import pandas as pd

data = [
    ["삼성","1000","2000"],
    ["현대","1100","3000"],
    ["LG","2000","500"],
    ["아모레","3500","6000"],
    ["naver","100","1500"],
]

index = ["031","059","033","045","023"]
colums = ["종목명","시가","종가"]

df = pd.DataFrame(data=data, index=index, columns=colums)
print(df)

# df[0]에러
# df["031"]에러
print(df["종목명"])

#판다스의 기준은 열이고 컬럼명을(기준) 입력해야한다
#아모레를 출력하고싶어
#print(df[4,0]) key에러
#print(df["종목명",["045"]]) key에러
#판다스는 행열이 아닌 열 행이다

#loc:인덱스를 기준으로 행 데이터 추출
#iloc:행번호를 기준으로 행 데이터 추출
print("===아모레===")
print(df.loc["045"])
# print(df.loc[3])  #key 에러
print(df.iloc[3])
print("====naver====")
print(df.loc["023"])
print(df.iloc[4])
print(df.iloc[-1])
print("====아모레 시가====")
print(df.loc["045"].loc["시가"])
print(df.loc["045"].iloc[1])
print(df.iloc[3].iloc[1])
print(df.iloc[3].loc["시가"])

print(df.loc["045"][1])
print(df.iloc[3][1])

print(df.loc["045"]["시가"])
print(df.iloc[3]["시가"])

print(df.loc["045","시가"])
print(df.iloc[3,1])

print("====== 아모레와 네이버의 시가뽑자")
print(df.iloc[3:5,1])
print(df.loc["045:023","시가"])
