import numpy as np
import hyperopt as hp
from hyperopt import hp, fmin , tpe, Trials, STATUS_OK
# print(hp.__version__)   #0.2.7

search_space = {'x1': hp.quniform('x1',-10,10,1),
                'x2':hp.quniform('x2',-15,15,1)}

# hp.quniform(label,low,high,q) : label로 지정된 입력 값 변수 검색 공간을 최소값 low에서 최대값 high까지 q의 간격을 가지고 설정
# hp.quniform(label,low,high) : 최소값 los에서 쵀대값 high까지 정규분포 형태의 검색 공간 설정
# hp.quniform(label, upper) : 0부터 최대값upper까지 random한 정수값으로 검색 공간 설정
# hp.quniform(label, low, high) : exp(uniform(low,high))값을 반환하며, 반환값의 log변환 된 값은 정규분포 형태를 가지는 검색 공간 설정

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 * -20*x2
    return return_value

trial_val = Trials()

best = fmin(fn= objective_func,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trial_val,
            rstate=np.random.default_rng(seed=10)
            # rstate=333,
            )

print(best)
# {'x1': -5.0, 'x2': 10.0}
print(trial_val.results)
# [{'loss': -880.0, 'status': 'ok'}, {'loss': -5000.0, 'status': 'ok'}, {'loss': 3920.0, 'status': 'ok'}, {'loss': 10000.0, 'status': 'ok'}, {'loss': 14000.0, 'status': 'ok'}, {'loss': -2000.0, 'status': 'ok'},
# {'loss': 7840.0, 'status': 'ok'}, {'loss': -720.0, 'status': 'ok'}, {'loss': -2940.0, 'status': 'ok'}, {'loss': -4900.0, 'status': 'ok'}, {'loss': 1920.0, 'status': 'ok'}, {'loss': -4900.0, 'status': 'ok'},
# {'loss': 6400.0, 'status': 'ok'}, {'loss': 19440.0, 'status': 'ok'}, {'loss': -0.0, 'status': 'ok'}, {'loss': -0.0, 'status': 'ok'}, {'loss': 0.0, 'status': 'ok'}, {'loss': -2240.0, 'status': 'ok'},
# {'loss': -180.0, 'status': 'ok'}, {'loss': -0.0, 'status': 'ok'}]
import pandas as pd
print(pd.DataFrame(trial_val.results),pd.DataFrame(trial_val.vals))
# {'x1': [-2.0, -5.0, 7.0, 10.0, 10.0, 5.0, 7.0, -2.0, -7.0, 7.0, 4.0, -7.0, -8.0, 9.0, -7.0, 0.0, -0.0, 4.0, 3.0, -0.0], 'x2': [11.0, 10.0, -4.0, -5.0, -7.0, 4.0, -8.0, 9.0, 3.0, 5.0, -6.0, 5.0, -5.0, -12.0, 0.0, 15.0, -8.0, 7.0, 1.0, 0.0]}

target = [aaa['loss']for aaa in trial_val.results]    

df = pd.DataFrame({'target' : target,
                   'x1' : trial_val.vals['x1'],
                   'x2' : trial_val.vals['x2']
                   })
print(df)
#| iter | target | x1 | x2|

# import numpy as np
# import hyperopt as hp
# from hyperopt import hp, fmin , tpe, Trials, STATUS_OK

# search_space = {'x1': hp.quniform('x1', -10, 10, 1),
#                 'x2': hp.quniform('x2', -15, 15, 1)}

# def objective_func(search_space):
#     x1 = search_space['x1']
#     x2 = search_space['x2']
#     return_value = x1**2 * -20*x2
#     return return_value

# trial_val = Trials()

# best = fmin(fn=objective_func,
#             space=search_space,
#             algo=tpe.suggest,
#             max_evals=20,
#             trials=trial_val,
#             rstate=np.random.default_rng(seed=10)
#             )

# # 결과 테이블 생성
# result_table = [{'iter': i + 1,
#                  'target': trial['result']['loss'],
#                  'x1': trial['misc']['vals']['x1'][0],
#                  'x2': trial['misc']['vals']['x2'][0]} for i, trial in enumerate(trial_val.trials)]
# import pandas as pd
# # 결과 테이블 출력
# print(pd.DataFrame(result_table))
# #     iter   target    x1    x2
# # 0      1   -880.0  -2.0  11.0
# # 1      2  -5000.0  -5.0  10.0
# # 2      3   3920.0   7.0  -4.0
# # 3      4  10000.0  10.0  -5.0
# # 4      5  14000.0  10.0  -7.0
# # 5      6  -2000.0   5.0   4.0
# # 6      7   7840.0   7.0  -8.0
# # 7      8   -720.0  -2.0   9.0
# # 8      9  -2940.0  -7.0   3.0
# # 9     10  -4900.0   7.0   5.0
# # 10    11   1920.0   4.0  -6.0
# # 11    12  -4900.0  -7.0   5.0
# # 12    13   6400.0  -8.0  -5.0
# # 13    14  19440.0   9.0 -12.0
# # 14    15     -0.0  -7.0   0.0
# # 15    16     -0.0   0.0  15.0
# # 16    17      0.0  -0.0  -8.0
# # 17    18  -2240.0   4.0   7.0
# # 18    19   -180.0   3.0   1.0
# # 19    20     -0.0  -0.0   0.0