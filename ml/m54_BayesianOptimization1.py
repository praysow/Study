param_bounds = {'x1':(-1,5),
                'x2':(0,4)}

def y_function(x1,x2):
    return -x1 **2 - (x2 -2) **2+10

#pip install BayesianOptimization

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f= y_function,
    pbounds=param_bounds,
    random_state=777
)
optimizer.maximize(init_points=5,
                   n_iter=20)
print(optimizer.max)
# |   iter    |  target   |    x1     |    x2     |
# -------------------------------------------------
# | 1         | 9.368     | -0.08402  | 1.209     |
# | 2         | 9.58      | -0.6278   | 1.839     |
# | 3         | -9.01     | 4.012     | 3.708     |
# | 4         | -2.456    | 3.362     | 3.074     |
# | 5         | 9.29      | 0.6152    | 2.576     |
# | 6         | -2.193    | 2.862     | 0.0       |
# | 7         | 5.0       | -1.0      | 4.0       |
# | 8         | 5.0       | -1.0      | 0.0       |
# | 9         | 4.957     | 1.021     | 4.0       |
# | 10        | 8.3       | 1.262     | 1.672     |
# | 11        | 9.969     | 0.1645    | 1.937     |
# | 12        | 9.558     | -0.3077   | 2.589     |
# | 13        | 8.81      | -1.0      | 2.436     |
# | 14        | 9.965     | -0.1773   | 2.059     |
# | 15        | 9.929     | -0.07471  | 1.745     |
# | 16        | 9.946     | 0.07649   | 2.219     |
# | 17        | 10.0      | -0.000957 | 1.997     |
# | 18        | 10.0      | 0.001859  | 2.006     |
# | 19        | 10.0      | -0.003861 | 1.989     |
# | 20        | 9.998     | 0.01054   | 2.038     |
# | 21        | 9.998     | -0.01215  | 1.962     |
# | 22        | 9.999     | 0.01185   | 2.032     |
# | 23        | 9.999     | -0.008843 | 1.968     |
# | 24        | 9.998     | -0.01797  | 2.042     |
# | 25        | 9.998     | 0.04345   | 2.017     |
# =================================================
# {'target': 9.999987959248532, 'params': {'x1': -0.0009570992627137529, 'x2': 1.9966646270868116}}