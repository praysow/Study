import pandas as pd

df = pd.DataFrame({'a' : [1,2,3,4,5],
                   'B' : [10,20,30,40,50],
                   'c' : [5,4,3,2,1],
                   'D' : [0.,1.,2.,3.,4.],
                   })

correlations = df.corr()
print(correlations)