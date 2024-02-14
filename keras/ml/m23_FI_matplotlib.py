import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1.데이터
datasets = load_iris()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,
                                                    random_state=1,
                                                    stratify=y
                                                    )

allAlgorithms = [
    ('XGBClassifier', XGBClassifier),
    ('GradientBoostingClassifier', GradientBoostingClassifier),
    ('DecisionTreeClassifier', DecisionTreeClassifier),
    ('RandomForestClassifier', RandomForestClassifier)
]
def plot_feature_importances_dataset(model, model_name):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    plt.title(model_name)


# 3. 모델 훈련 및 평가
for name, algorithm in allAlgorithms:
    model = algorithm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    plt.subplot(2, 2, 1)
    plot_feature_importances_dataset(model,'XGBClassifier')
    plt.subplot(2, 2, 2)
    plot_feature_importances_dataset(model, 'GradientBoostingClassifier')
    plt.subplot(2, 2, 3)
    plot_feature_importances_dataset(model, 'DecisionTreeClassifier')
    plt.subplot(2, 2, 4)
    plot_feature_importances_dataset(model, 'RandomForestClassifier')
    print(name, '의 정확도:', acc)

plt.tight_layout()
plt.show()


