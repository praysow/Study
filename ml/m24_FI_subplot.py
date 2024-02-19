from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
import numpy as np
from xgboost import XGBClassifier

class CustomXGBClassifier(XGBClassifier):
    def __str__(self):
        return "XGBClassifier()"

# 1. 데이터
# x, y = datasets = load_iris(return_X_y=True)
datasets = load_iris()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=13, stratify=y) 


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

random_state=42
#모델구성
models = [DecisionTreeClassifier(random_state=random_state), RandomForestClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state), CustomXGBClassifier(random_state=random_state)]
for model in models :

    # 컴파일, 훈련
    model.fit(x_train, y_train)

    # 평가, 예측
    from sklearn.metrics import accuracy_score
    results = model.score(x_test, y_test)
    print(f"[{type(model).__name__}] model.score : ", results)#정확도:  1.0

    x_predict = model.predict(x_test)
    # print(x_predict)
    acc_score = accuracy_score(y_test, x_predict)
    print(f"[{type(model).__name__}] model accuracy_score : ", acc_score)

    print(type(model).__name__ ,":", model.feature_importances_) #중요도/ 낮다고 지우는건 위험함. 성능이 확 떨어짐
    
import matplotlib.pyplot as plt
plt.figure(figsize=(11, 5))
bar_colors = ['red', 'blue', 'green', 'purple']
for idx, model in enumerate(models) :
    import numpy as np
    def plot_feature_importances_datasest(model):
        n_features = datasets.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align="center",color = bar_colors[idx]) #막대그래프 설정
        plt.yticks(np.arange(n_features), datasets.feature_names) #눈금값 설정
        plt.xlabel("Feature Importances") #xlabel
        plt.ylabel("Features") #ylabel
        top,bottom = plt.ylim(-1, n_features) #y축 제한 
        print(top, bottom)
        plt.title(model, pad=15)
    plt.subplot(2,2,idx+1)
    plt.subplots_adjust(left=0.2,bottom=0.1, wspace=1, hspace=1)
    plot_feature_importances_datasest(model)
plt.show()