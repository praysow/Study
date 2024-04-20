from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import pandas as pd

def image_scaler(x_train:np.array, x_test:np.array, scaler:str):
    '''
    image를 scaling하는 함수입니다
    반환값은 x_train, x_test입니다
    scaler 값은 minmax, standard, robust중 하나로 해주세요
    '''
    xtr0, xtr1, xtr2, xtr3 = (0, 0, 0, 0)
    if(len(x_train.shape)==3):
        xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    else:
        xtr0, xtr1, xtr2, xtr3 = (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3])
        xt0, xt1, xt2, xt3 = (x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3])

    x_train = x_train.reshape(xtr0, xtr1*xtr2*xtr3)
    x_test = x_test.reshape(xt0, xt1*xt2*xt3)

    if scaler == 'minmax':
        minmax = MinMaxScaler().fit(x_train)
        x_train = minmax.transform(x_train)
        x_test = minmax.transform(x_test)
    elif scaler == 'standard':
        standard = StandardScaler().fit(x_train)
        x_train = standard.transform(x_train)
        x_test = standard.transform(x_test)
    elif scaler == 'robust':
        robust = RobustScaler().fit(x_train)
        x_train = robust.transform(x_train)
        x_test = robust.transform(x_test)
    else:
        print(f"set wrong scaler({scaler}), set by 'minmax' or 'stardard' or 'robust'.")

    x_train = x_train.reshape(xtr0, xtr1, xtr2, xtr3)
    x_test = x_test.reshape(xt0, xt1, xt2, xt3)
    
    return x_train, x_test


def merge_image(img_iter, fail_stop=False):
    '''
    IDG를 돌려 나눠진 이미지데이터를 병합해주는 함수입니다
    argments:
        img_iter : ImageDataGenerator's iterator
        fail_stop: True면 예외 발생시 함수를 중지시킵니다
    returns:
        data, label
    '''
    x = []
    y = []
    failed_i = []
    
    for i in range(len(img_iter)):
        try:
            xy = img_iter.next()
            new_x = np.array(xy[0])
            new_y = np.array(xy[1])
            if i == 0:                  #바로 병합시키려 하면 shape가 동일하지않다는 오류가 나기에 최초 1회는 그대로 대입
                x = new_x
                y = new_y
                continue
            
            if len(new_y.shape) == 1:   #만약 new_y.shape = (N,) 형태라면, 즉 이진분류라면
                x = np.vstack([x,new_x])
                y = np.hstack([y,new_y])
            else:                       #이진분류가 아니니 다중분류라면
                x = np.vstack([x,new_x])
                y = np.vstack([y,new_y])
                
        except Exception as e:
            print("faied i: ",i)
            failed_i.append(i)
            if fail_stop:
                raise
                
        print("i: ",i)
        print(f"{x.shape=}\n{y.shape=}")    
        
    print("failed i list: ",failed_i)
    return x, y

def split_x(dataset, time_step:int):
    '''
    시계열데이터를 time_step만큼 잘라주는 함수입니다
    arguments
        dataset  : arraylike
        time_step: int
    return
        np.array
    '''
    result = []
    num = len(dataset) - time_step + 1
    if num <= 0: #자르는 것이 불가능한 time_step이 들어온 경우
        raise Exception(f"time_step:{time_step}이 너무 큽니다")
    
    for i in range(num):
        result.append(dataset[i : i+time_step])
        
    
    result = np.array(result)    
    
    if len(result.shape) == 1: #y같이 벡터형태인 경우
        result = result.reshape(1,result.shape[0],1)
    if len(result.shape) == 2: #feature가 1개라서 결과가 2차원인 경우
        result = result.reshape(result.shape[0],result.shape[1],1)
    
    return result

def split_xy(dataset:pd.DataFrame,time_step,y_col='None'):
    '''
    아직 시계열 데이터 등에 대한건 더 개선 필요
    '''
    dataset = dataset.astype(np.float32)
    dataset = dataset.to_numpy()
    result = np.array([])
    result_y = []
    if y_col == 'None':
        time_step += 1  # y까지 포함해서 잘라야하기에 +1
    else:
        num += 1        # y들어갈거 생각하면 1칸 비워줘야하기에

    num = len(dataset) - time_step + 1
    if num <= 0: # 자르는 것이 불가능한 time_step이 들어온 경우
        raise Exception(f"time_step:{time_step}이 너무 큽니다")
    # print(num)
    for idx, i in enumerate(range(num)):
        if idx == 0:
            result = np.array([dataset[i : i+time_step]])
            continue
        # print(f"{idx}번째 result: ",result.shape)
        result = np.concatenate([result, np.array([dataset[i : i+time_step]])],axis=0)
    
    # result = np.array(result)    
    
    if len(result.shape) == 2: # feature가 1개라서 결과가 2차원인 경우
        result = result.reshape(result.shape[0],result.shape[1],1)
    
    
    
    if y_col == 'None':
        return result[:,:-1,:] , result[:,-1,0]
    else:
        pass
    
acc = 1.0