import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# 데이터 불러오기
path = "c:/_data/dacon/soduc/"
train = pd.read_csv(path+'train.csv', index_col=0)
test = pd.read_csv(path+'test.csv', index_col=0)
sample = pd.read_csv(path+'sample_submission.csv')

# 피처와 타겟 분리
x = train.drop(['Income','Gains','Losses','Dividends'], axis=1)
y = train['Income']
test = test.drop(['Gains','Losses','Dividends'], axis=1)
lb = LabelEncoder()

# 라벨 인코딩할 열 목록
columns_to_encode = ['Gender','Education_Status','Employment_Status','Industry_Status','Occupation_Status','Race','Hispanic_Origin','Martial_Status','Household_Status','Household_Summary','Citizenship','Birth_Country','Birth_Country (Father)','Birth_Country (Mother)','Tax_Status','Income_Status']

# 데이터프레임 x의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(x[column])
    x[column] = lb.transform(x[column])

# 데이터프레임 test_csv의 열에 대해 라벨 인코딩 수행
for column in columns_to_encode:
    lb.fit(test[column])
    test[column] = lb.transform(test[column])
    
# 데이터 스케일링
scaler = StandardScaler()
x = scaler.fit_transform(x)
test = scaler.transform(test)

rmses = []
random_states = []

for r in range(101, 201):
    # 훈련 데이터와 검증 데이터 분리
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.9, random_state=r)

    # XGBoost 모델 학습
    xgb_params = {'learning_rate': 0.2218036245351803,
                'n_estimators': 199,
                'max_depth': 3,
                'min_child_weight': 0.07709868781803283,
                'subsample': 0.80309973945344,
                'colsample_bytree': 0.9254025887963853,
                'gamma': 6.628562492458777e-08,
                'reg_alpha': 0.012998871754325427,
                'reg_lambda': 0.10637051171111844}

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50, verbose=100)

    # 검증 데이터 예측
    y_pred_val = model.predict(x_val)

    # 검증 데이터 RMSE 계산
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    print(f"Validation RMSE with random_state={r}: {rmse_val}")

    # 테스트 데이터 예측 및 저장
    y_pred_test = model.predict(test)
    sample['Income'] = y_pred_test
    sample.to_csv(f"c:/_data/dacon/soduc/csv/money_xgboost_{r}.csv", index=False)

    rmses.append(rmse_val)
    random_states.append(r)

# 최종 결과 출력
for r, rmse in zip(random_states, rmses):
    print(f"Random_state={r}: Validation RMSE={rmse}")
    
'''
Random_state=1: Validation RMSE=582.4980697733463
Random_state=2: Validation RMSE=560.877884503738
Random_state=3: Validation RMSE=633.466168189024
Random_state=4: Validation RMSE=612.064307730473
Random_state=5: Validation RMSE=641.6100134880933
Random_state=6: Validation RMSE=582.8089532999796
Random_state=7: Validation RMSE=555.07409301465
Random_state=8: Validation RMSE=552.348350847527
Random_state=9: Validation RMSE=614.860306833884
Random_state=10: Validation RMSE=570.5046433054523
Random_state=11: Validation RMSE=629.0556690878182
Random_state=12: Validation RMSE=616.4618488114027
Random_state=13: Validation RMSE=545.5304991146946
Random_state=14: Validation RMSE=639.4157082702636
Random_state=15: Validation RMSE=667.9982627949122
Random_state=16: Validation RMSE=630.3076956079539
Random_state=17: Validation RMSE=625.6777249119991
Random_state=18: Validation RMSE=516.3583199725235
Random_state=19: Validation RMSE=527.9052752640092
Random_state=20: Validation RMSE=561.4590348850426
Random_state=21: Validation RMSE=604.1635567065385
Random_state=22: Validation RMSE=630.9319219705569
Random_state=23: Validation RMSE=531.8875104256482
Random_state=24: Validation RMSE=611.77506839572
Random_state=25: Validation RMSE=645.5599040191439
Random_state=26: Validation RMSE=670.3480073037922
Random_state=27: Validation RMSE=591.2745914872352
Random_state=28: Validation RMSE=627.056161033452
Random_state=29: Validation RMSE=631.6278797012186
Random_state=30: Validation RMSE=606.5945727722493
Random_state=31: Validation RMSE=555.1375601102094
Random_state=32: Validation RMSE=584.1375248468191
Random_state=33: Validation RMSE=581.4005898800402
Random_state=34: Validation RMSE=587.8129998487764
Random_state=35: Validation RMSE=582.6278945643169
Random_state=36: Validation RMSE=692.9170210091238
Random_state=37: Validation RMSE=509.2791637700279
Random_state=38: Validation RMSE=528.3234061389828
Random_state=39: Validation RMSE=652.9308783631313
Random_state=40: Validation RMSE=582.3907960479897
Random_state=41: Validation RMSE=650.3796403974335
Random_state=42: Validation RMSE=609.202190018697
Random_state=43: Validation RMSE=655.3679966818727
Random_state=44: Validation RMSE=551.3788388417955
Random_state=45: Validation RMSE=670.7727470809924
Random_state=46: Validation RMSE=630.7078773053165
Random_state=47: Validation RMSE=565.2082192572877
Random_state=48: Validation RMSE=550.9516385654498
Random_state=49: Validation RMSE=561.8895890483371
Random_state=50: Validation RMSE=564.0530722781866
Random_state=51: Validation RMSE=624.7660060198419
Random_state=52: Validation RMSE=519.3147724301665
Random_state=53: Validation RMSE=606.1601347693485
Random_state=54: Validation RMSE=533.0004394449896
Random_state=55: Validation RMSE=573.2953591533684
Random_state=56: Validation RMSE=647.4289728641152
Random_state=57: Validation RMSE=599.7157817933635
Random_state=58: Validation RMSE=650.096862685792
Random_state=59: Validation RMSE=605.8347906802938
Random_state=60: Validation RMSE=585.5287280760415
Random_state=61: Validation RMSE=587.7610200084595
Random_state=62: Validation RMSE=632.896385294525
Random_state=63: Validation RMSE=614.8740627670363
Random_state=64: Validation RMSE=556.8407487994594
Random_state=65: Validation RMSE=514.3418500822435
Random_state=66: Validation RMSE=592.3131489293877
Random_state=67: Validation RMSE=580.5828016672974
Random_state=68: Validation RMSE=583.862209947304
Random_state=69: Validation RMSE=565.5386361280835
Random_state=70: Validation RMSE=583.9287600211097
Random_state=71: Validation RMSE=614.236228449493
Random_state=72: Validation RMSE=637.5454845612728
Random_state=73: Validation RMSE=656.1664689650959
Random_state=74: Validation RMSE=546.3087073359529
Random_state=75: Validation RMSE=515.7085626834034
Random_state=76: Validation RMSE=612.5656847636566
Random_state=77: Validation RMSE=506.6683414645201
Random_state=78: Validation RMSE=548.3134344378676
Random_state=79: Validation RMSE=597.437028768058
Random_state=80: Validation RMSE=530.2233975130557
Random_state=81: Validation RMSE=604.8556190834022
Random_state=82: Validation RMSE=537.3642338721268
Random_state=83: Validation RMSE=573.1616375102532
Random_state=84: Validation RMSE=547.0809447048472
Random_state=85: Validation RMSE=613.5044514656165
Random_state=86: Validation RMSE=636.3834619596536
Random_state=87: Validation RMSE=560.7988950090167
Random_state=88: Validation RMSE=636.6139084687445
Random_state=89: Validation RMSE=544.1160930401455
Random_state=90: Validation RMSE=534.6009411060542
Random_state=91: Validation RMSE=551.7669343046223
Random_state=92: Validation RMSE=543.5683874992352
Random_state=93: Validation RMSE=536.2043751559021
Random_state=94: Validation RMSE=622.9361964765978
Random_state=95: Validation RMSE=602.5472032739013
Random_state=96: Validation RMSE=598.1678049952359
Random_state=97: Validation RMSE=630.7637732965178
Random_state=98: Validation RMSE=554.4247586289299
Random_state=99: Validation RMSE=647.2668688701765
Random_state=100: Validation RMSE=612.1365775499777
Random_state=101: Validation RMSE=631.7395333066821
Random_state=102: Validation RMSE=611.8590154012832
Random_state=103: Validation RMSE=557.0620206912704
Random_state=104: Validation RMSE=594.2531219314383
Random_state=105: Validation RMSE=582.4380690766303
Random_state=106: Validation RMSE=563.7964879628106
Random_state=107: Validation RMSE=523.3042082306941
Random_state=108: Validation RMSE=643.3137955793133
Random_state=109: Validation RMSE=593.9608607802608
Random_state=110: Validation RMSE=660.5298300290557
Random_state=111: Validation RMSE=600.476964511105
Random_state=112: Validation RMSE=599.5990212119199
Random_state=113: Validation RMSE=608.2439304353572
Random_state=114: Validation RMSE=681.2276379357276
Random_state=115: Validation RMSE=609.5731649700443
Random_state=116: Validation RMSE=607.5631720703719
Random_state=117: Validation RMSE=578.5162195352559
Random_state=118: Validation RMSE=521.7320540103884
Random_state=119: Validation RMSE=556.5046849061736
Random_state=120: Validation RMSE=550.7376098187082
Random_state=121: Validation RMSE=629.7202135808849
Random_state=122: Validation RMSE=613.5425692254051
Random_state=123: Validation RMSE=619.8451494371208
Random_state=124: Validation RMSE=519.6906575502983
Random_state=125: Validation RMSE=659.3886517946937
Random_state=126: Validation RMSE=623.0692251268204
Random_state=127: Validation RMSE=535.7401194535632
Random_state=128: Validation RMSE=645.3908099101125
Random_state=129: Validation RMSE=585.0983176723722
Random_state=130: Validation RMSE=617.4700610496458
Random_state=131: Validation RMSE=521.0502409412657
Random_state=132: Validation RMSE=643.1281236706901
Random_state=133: Validation RMSE=614.145372371114
Random_state=134: Validation RMSE=555.4379069603882
Random_state=135: Validation RMSE=544.8872671615433
Random_state=136: Validation RMSE=587.4956895830712
Random_state=137: Validation RMSE=618.4278486039589
Random_state=138: Validation RMSE=558.4516471057229
Random_state=139: Validation RMSE=592.6619680300666
Random_state=140: Validation RMSE=591.8758157996454
Random_state=141: Validation RMSE=621.9084446134491
Random_state=142: Validation RMSE=594.1238000639374
Random_state=143: Validation RMSE=636.6176886982611
Random_state=144: Validation RMSE=579.5806238890024
Random_state=145: Validation RMSE=659.0640315470285
Random_state=146: Validation RMSE=595.5154071674369
Random_state=147: Validation RMSE=626.1394112543541
Random_state=148: Validation RMSE=589.3857096983046
Random_state=149: Validation RMSE=588.5467515611081
Random_state=150: Validation RMSE=599.947228589321
Random_state=151: Validation RMSE=717.5641559733867
Random_state=152: Validation RMSE=589.5594984920091
Random_state=153: Validation RMSE=603.0276184804003
Random_state=154: Validation RMSE=548.3128944942022
Random_state=155: Validation RMSE=620.2828250100343
Random_state=156: Validation RMSE=636.353436230771
Random_state=157: Validation RMSE=646.0827345184415
Random_state=158: Validation RMSE=579.4081891369266
Random_state=159: Validation RMSE=609.7977253181579
Random_state=160: Validation RMSE=672.4914573603538
Random_state=161: Validation RMSE=603.686785404789
Random_state=162: Validation RMSE=573.6468520700749
Random_state=163: Validation RMSE=649.4665199910427
Random_state=164: Validation RMSE=611.7672661112862
Random_state=165: Validation RMSE=667.9909618915685
Random_state=166: Validation RMSE=651.7593831140931
Random_state=167: Validation RMSE=547.0394118405947
Random_state=168: Validation RMSE=661.1925965866508
Random_state=169: Validation RMSE=644.4029804983358
Random_state=170: Validation RMSE=618.1479674482683
Random_state=171: Validation RMSE=655.9308757774355
Random_state=172: Validation RMSE=577.2458935706561
Random_state=173: Validation RMSE=592.3891077248109
Random_state=174: Validation RMSE=561.0698208774252
Random_state=175: Validation RMSE=688.6183096609583
Random_state=176: Validation RMSE=645.3725376496443
Random_state=177: Validation RMSE=562.7729036578464
Random_state=178: Validation RMSE=681.0561041543152
Random_state=179: Validation RMSE=599.2396524378779
Random_state=180: Validation RMSE=633.5509885179989
Random_state=181: Validation RMSE=607.5694419204245
Random_state=182: Validation RMSE=571.9393525648848
Random_state=183: Validation RMSE=549.546895363865
Random_state=184: Validation RMSE=633.8706697395013
Random_state=185: Validation RMSE=567.0249668133957
Random_state=186: Validation RMSE=585.3101648903977
Random_state=187: Validation RMSE=590.3943715800131
Random_state=188: Validation RMSE=538.6311926444787
Random_state=189: Validation RMSE=620.4272849883553
Random_state=190: Validation RMSE=583.1779683719699
Random_state=191: Validation RMSE=580.3363106701092
Random_state=192: Validation RMSE=609.8942494260567
Random_state=193: Validation RMSE=584.7912266622197
Random_state=194: Validation RMSE=667.6726422817716
Random_state=195: Validation RMSE=558.897914942648
Random_state=196: Validation RMSE=581.7697773620428
Random_state=197: Validation RMSE=585.5594433131961
Random_state=198: Validation RMSE=600.072083812996
Random_state=199: Validation RMSE=558.9325112821388
Random_state=200: Validation RMSE=527.4870253671552
'''
