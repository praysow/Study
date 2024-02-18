#pca를 통해 0.95이상인 n_components는 몇개?
#0.95 이상
#0.99이상
#0.999이상
#1.0일때 몇개?
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

(x_train,y_train),(x_test,y_test) = mnist.load_data()
# print(x_train.shape,x_test.shape)   #(60000, 28, 28) (10000, 28, 28)

# x=np.append(x_train,x_test,axis=0)
x=np.concatenate([x_train,x_test],axis=0)
y=np.concatenate([y_train,y_test],axis=0)

x = x.reshape(70000,28*28)
# print(x.shape)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)
# print(x.shape)

pca = PCA(n_components=784)
x = pca.fit_transform(x)
evr = pca.explained_variance_ratio_
cumsum = np.cumsum(evr)
# print(cumsum)

print(np.argmax(cumsum >= 0.95)+1)
print(np.argmax(cumsum >= 0.99)+1)
print(np.argmax(cumsum >= 0.999)+1)
print(np.argmax(cumsum >= 1.0)+1)





# pca = PCA(n_components=784)
# x= pca.fit_transform(x)
# evr = pca.explained_variance_ratio_
# # print(evr)
# # print(sum(evr))
# evr_cumsum = np.cumsum(evr)      #누적합
# # 0.95 이상인 n_components 개수 계산
# n_components_095 = np.argmax(evr_cumsum >= 0.95) + 1
# print("0.95 이상인 n_components 개수:", n_components_095)

# # 0.99 이상인 n_components 개수 계산
# n_components_099 = np.argmax(evr_cumsum >= 0.99) + 1
# print("0.99 이상인 n_components 개수:", n_components_099)

# # 0.999 이상인 n_components 개수 계산
# n_components_0999 = np.argmax(evr_cumsum >= 0.999) + 1
# print("0.999 이상인 n_components 개수:", n_components_0999)

# # 1.0일 때의 n_components 개수는 전체 차원(784)과 동일합니다.
# n_components_1 = 784
# print("1.0일 때의 n_components 개수:", n_components_1)

'''
0.95 이상인 n_components 개수: 332
0.99 이상인 n_components 개수: 544
0.999 이상인 n_components 개수: 683
1.0일 때의 n_components 개수: 784
'''

# print(x.shape)
# x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=3,shuffle=True,stratify=y)
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # x_train = scaler.fit_transform(x_train)
# # x_test = scaler.transform(x_test)
# # print(x_train.shape,x_test.shape)
# # ra = np.arange(1, min(x_train.shape) + 1)
# ra = [784]

# for n_components in ra:
#     pca = PCA(n_components=n_components)
#     x_train_pca = pca.fit_transform(x_train)
#     x_test_pca = pca.transform(x_test)
#     evr = pca.explained_variance_ratio_

#     # PCA 모델 학습 및 평가
#     model = RandomForestRegressor()
#     model.fit(x_train_pca, y_train)
#     acc = model.score(x_test_pca, y_test)
#     print(f'n_components={n_components}의 정확도:', acc)
#     evr_cumsum = np.cumsum(evr)      #누적합
#     n_components_095 = np.argmax(evr_cumsum >= 0.95) + 1
#     print("0.95 이상인 n_components 개수:", n_components_095)

#     # 0.99 이상인 n_components 찾기
#     n_components_099 = np.argmax(evr_cumsum >= 0.99) + 1
#     print("0.99 이상인 n_components 개수:", n_components_099)

#     # 0.999 이상인 n_components 찾기
#     n_components_0999 = np.argmax(evr_cumsum >= 0.999) + 1
#     print("0.999 이상인 n_components 개수:", n_components_0999)

#     # 1.0일 때의 n_components 개수는 전체 차원(784)과 동일합니다.
#     n_components_1 = 784
#     print("1.0일 때의 n_components 개수:", n_components_1)
# import matplotlib.pyplot as plt
# plt.plot(evr_cumsum)
# plt.grid()
# # plt.show()