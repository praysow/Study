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
# print(df)
#        종목명    시가    종가
# 031    삼성  1000  2000
# 059    현대  1100  3000
# 033    LG    2000  500
# 045    아모레3500  6000
# 023    naver 100  1500
# df["시가"] = df["시가"].astype(int)
# df["종가"] = df["종가"].astype(int)

# # 시가가 1100 이상인 행을 필터링하여 출력
# si = df[df["시가"] >= 1100]
# jong = df[df["종가"] >= 1100]["종가"]
# print("==시가==")
# print(si)
# print("==종가==")
# print(jong)

print(df[df["시가"] >= '1100']['종가'])
# print(df[df["시가"] >= '1100'][2])#에러
print(df.loc[df["시가"] >= '1100']['종가'])
print(df.loc[df["시가"] >= '1100','종가'])


