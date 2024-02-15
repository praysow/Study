#xgboost와 그리드서치,랜덤서치, 하빙서치중 하나를 사용

#n_jobs = -1 

    #tree_method='gpu_hist',
    #predictor='gpu_predictor',
    #gpu_id=0
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense
import time
from keras.utils import to_categorical
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

x = x.reshape(70000, 28*28)
lda = LinearDiscriminantAnalysis(n_components=1)
x= lda.fit_transform(x,y)
# scaler = StandardScaler()
# x = scaler.fit_transform(x)


from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
n_split = 5
kfold = KFold(n_splits=n_split,shuffle=True, random_state=100)
parameters = {
    'n_estimators': [100, 200, 90, 110], 
    'max_depth': [4, 5, 6], 
    'learning_rate': [0.1, 0.3, 0.001],
    'colsample_bytree': [0.6, 0.9, 1],
    'colsample_bylevel': [0.6, 0.7, 0.9],
    # 'tree_method': ['gpu_hist'],
    # 'predictor': ['gpu_predictor'],
    # 'gpu_id': [0]
}

st = time.time()

n_components_list = [1]
results = {}
from keras.callbacks import EarlyStopping,ModelCheckpoint
for n_components in n_components_list:
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(x)
    evr = pca.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    print(f"Explained Variance Ratio for {n_components} components:", cumsum[-1])

    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, train_size=0.9, random_state=3, shuffle=True, stratify=y)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model3 = HalvingRandomSearchCV(XGBClassifier(),parameters, cv = kfold,verbose=1,refit=True,n_jobs=-1,random_state=6,factor=3.5,min_resources=10000)
    model3.fit(x_train,y_train)

    # 4.결과예측
    result3 = model3.score(x_test,y_test)
    y_predict3 = model3.predict(x_test)

    print("acc :", result3)
et = time.time()    

for n_components, result in results.items():
    print(f"Results for {n_components} components:")
    print("Loss:", result["loss"])
    print("Accuracy:", result["accuracy"])
    print("Elapsed Time:", result["elapsed_time"])
print("시간:",et-st)
'''
acc : 0.18657142857142858
시간: 5.20328426361084
'''