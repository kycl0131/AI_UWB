

def DataOpenandSave(path):
    import os
    import natsort
    import torch
    import pandas as pd
    import struct
    import numpy as np
    import matplotlib.pyplot as plt

    # data file path
    path  = path
    
    def list_files(directory):
        """
        directory 내의 파일 위치명 리스트를 반환하는 함수
        """
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    file_list = list_files(path)
    # print(file_list)

    file_list_star1 = [x for x in file_list if x.startswith(path+'/beam')]
    file_list_star1 = list_files(file_list_star1[0])
    print(file_list_star1)
    file_list_star1 = natsort.natsorted(file_list_star1)
    file_list_star1 = [x for x in file_list_star1 if x.startswith(path+'/beam_test'+'/xethru_datafloat_')]

    file_list_star2 = [x for x in file_list if x.startswith(path+'/elec')]
    file_list_star2 = list_files(file_list_star2[0])
    file_list_star2 = natsort.natsorted(file_list_star2)
    file_list_star2 = [x for x in file_list_star2 if x.startswith(path+'/elecle_test'+'/xethru_datafloat_')]

    file_list_star3 = [x for x in file_list if x.startswith(path+'/person')]
    file_list_star3 = list_files(file_list_star3[0])
    file_list_star3 = natsort.natsorted(file_list_star3)
    file_list_star3 = [x for x in file_list_star3 if x.startswith(path+'/person_test'+'/xethru_datafloat_')]


    file_list_star4 = [x for x in file_list if x.startswith(path+'/yellowmoby')]
    file_list_star4 = list_files(file_list_star4[0])
    file_list_star4 = natsort.natsorted(file_list_star4)
    file_list_star4 = [x for x in file_list_star4 if x.startswith(path+'/yellowmoby_test'+'/xethru_datafloat_')]


    DataClass = [x for lst in [file_list_star1, file_list_star2, file_list_star3,file_list_star4] for x in lst]

    for name in DataClass:
        print(f'dksjdkjskdjksdjsds{name}')
        
        # datafloat file is 4 byte float type 
        with open(name, 'rb') as file:
            data = file.read()
        # save opened data in rawData list
        rawData = []
        for i in range(0, len(data), 4):  # assuming each float is 4 bytes
            rawData.append(struct.unpack('f', data[i:i+4]))
        #invert to numpy to reshape datasize    
        rawData = np.array(rawData)

        # set threshold to max value of rawData 
        # (The original data is one-dimensional amplitude data representing intensity over distance, which is measured at the same time interval and concatenated.)
        #  ex. (100000,1) -> (-1,1512) 1512 is example of one data size
        threshold = 0.0025 # max(rawData).round(3).item()  
        tresh_idx = []  # #(61)- > 1512 ->(1573)-> 1512-> (3085)
        i = 0
        while i < len(rawData):
            if rawData[i][0] >= threshold:
                tresh_idx.append(i)
                # print(i)
                i += 1340
            else:
                i += 1
        
        print(f'threshold:{len(tresh_idx)}')        
        rawData_reshaped = rawData.reshape(1,33,-1)
        # print(rawData_reshaped.shape)
        #labeling

        if name in file_list_star1:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 0)
        
        elif name in file_list_star2:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 1)
        
        elif name in file_list_star3:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 2)
        elif name in file_list_star4:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 3)
        
        
        # print(f'X shape{X.shape}, Y shape{Y.shape}')
        
        if name == DataClass[0]:
            print('초기화')
            Y = label
            X = rawData_reshaped
            
            # print(f'X shape{X.shape}, Y shape{Y.shape}')
        else: 
            # print(X.shape, rawData_reshaped.shape)
            Y = np.concatenate((Y,label),axis=0)
            # print(name)
            X= np.concatenate((X,rawData_reshaped),axis=0)
            

    ##plot this data
    # plt.plot(rawData_reshaped[:],alpha=1)
    # plt.show()
    # x = torch.unsqueeze(torch.from_numpy(X),dim=1)
    # y = torch.from_numpy(Y)


    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler,MaxAbsScaler
    import os
    import natsort
    import chardet


    # hyperparameter
    # batch_size = 128
    # learning_rate = 0.01
    # epochs = 200
    datasize = 1528 #1512
    num_range_max = 1078 # 1512




    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from torch.autograd import Variable
    ########
    ######
    ###
    rangevec = np.arange(0,1078,1078/num_range_max) # detect range 0~10m -> 0~ 1000
    start = 150
    end =600
    start_id= np.where(np.floor(rangevec) == start)

    # 가장 가까운 값의 인덱스를 찾기 위해 절댓값 차이 계산
    diff_end = np.abs(rangevec - end)

    # 가장 작은 차이를 갖는 인덱스 반환
    closest_index = np.argmin(diff_end)

    # end_id = np.where(np.round(rangevec) == end)
    start = start_id[0][0]
    # end = end_id[0][0]
    end = closest_index

    # use gpu(apple m1)
    # device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)


    #split train, val, test
    from sklearn.model_selection import train_test_split
    # 33 = 원ㅐ len(tresh_idx)
    x_test = X[:,:,start:end].reshape(-1,33,end-start)
    y_test = Y
    # y = F.one_hot(torch.tensor(y, dtype=torch.int64)).numpy()


    # zero padding 1d
    x_test = np.pad(x_test, ((0,0),(0,0),(start, datasize - end)), 'constant', constant_values=0)
    print(x_test.shape)



    ## Train




    import pandas as pd
    import struct
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import natsort
    import torch


    # data file path
    path = '/home/yunkwan/pythonProject/temp/AI_UWB/Data_raw/Record/train'

    def list_files(directory):
        """
        directory 내의 파일 위치명 리스트를 반환하는 함수
        """
        return [os.path.join(directory, f) for f in os.listdir(directory)]

    file_list = list_files(path)

    # file_list = natsort.natsorted(file_list)



    file_list_star1 = [x for x in file_list if x.startswith(path+'/beam')]
    file_list_star1 = list_files(file_list_star1[0])
    print(file_list_star1)
    file_list_star1 = natsort.natsorted(file_list_star1)
    file_list_star1 = [x for x in file_list_star1 if x.startswith(path+'/beam'+'/xethru_datafloat_')]

    file_list_star2 = [x for x in file_list if x.startswith(path+'/elec')]
    file_list_star2 = list_files(file_list_star2[0])
    file_list_star2 = natsort.natsorted(file_list_star2)
    file_list_star2 = [x for x in file_list_star2 if x.startswith(path+'/elecle'+'/xethru_datafloat_')]

    file_list_star3 = [x for x in file_list if x.startswith(path+'/person')]
    file_list_star3 = list_files(file_list_star3[0])
    file_list_star3 = natsort.natsorted(file_list_star3)
    file_list_star3 = [x for x in file_list_star3 if x.startswith(path+'/person'+'/xethru_datafloat_')]


    file_list_star4 = [x for x in file_list if x.startswith(path+'/yellowmoby')]
    file_list_star4 = list_files(file_list_star4[0])
    file_list_star4 = natsort.natsorted(file_list_star4)
    file_list_star4 = [x for x in file_list_star4 if x.startswith(path+'/yellowmoby'+'/xethru_datafloat_')]


    DataClass = [x for lst in [file_list_star1, file_list_star2, file_list_star3 ,file_list_star4] for x in lst]


    # DataClass = [x for lst in [file_list_water, file_list_star ] for x in lst]#[file_list_water,file_list_star]

    # Y = np.empty((0, 1))
    # X = np.empty((0, 880)) 
    # open file and read data
    # DataClass[0]

    for name in DataClass:
        print(f'dksjdkjskdjksdjsds{name}')
        
        # datafloat file is 4 byte float type 
        with open(name, 'rb') as file:
            data = file.read()
        # save opened data in rawData list
        rawData = []
        for i in range(0, len(data), 4):  # assuming each float is 4 bytes
            rawData.append(struct.unpack('f', data[i:i+4]))
        #invert to numpy to reshape datasize    
        rawData = np.array(rawData)

        # set threshold to max value of rawData 
        # (The original data is one-dimensional amplitude data representing intensity over distance, which is measured at the same time interval and concatenated.)
        #  ex. (100000,1) -> (-1,1512) 1512 is example of one data size
        threshold =0.0005 # max(rawData).round(3).item()  
        tresh_idx = []  # #(61)- > 1512 ->(1573)-> 1512-> (3085)
        i = 0
        while i < len(rawData):
            if rawData[i][0] >= threshold:
                tresh_idx.append(i)
                # print(i)
                i += 1340
            else:
                i += 1
        
        print(f'threshold:{len(tresh_idx)}')        
        rawData_reshaped = rawData.reshape(1,33,-1)
        # print(rawData_reshaped.shape)
        #labeling
        if name in file_list_star1:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 0)
        
        elif name in file_list_star2:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 1)
        
        elif name in file_list_star3:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 2)
        elif name in file_list_star4:
            # print('1')
            label = np.full((rawData_reshaped.shape[0],1 ), 3)
        
        
        # print(f'X shape{X.shape}, Y shape{Y.shape}')
        
        if name == DataClass[0]:
            print('초기화')
            Y = label
            X = rawData_reshaped
            
            # print(f'X shape{X.shape}, Y shape{Y.shape}')
        else: 
            # print(X.shape, rawData_reshaped.shape)
            Y = np.concatenate((Y,label),axis=0)
            # print(name)
            X= np.concatenate((X,rawData_reshaped),axis=0)
            


    np.save('/home/yunkwan/pythonProject/temp/AI_UWB/Data_processed/x', X)
    np.save('/home/yunkwan/pythonProject/temp/AI_UWB/Data_processed/y', Y)
    np.save('/home/yunkwan/pythonProject/temp/AI_UWB/Data_processed/x_test', x_test)
    np.save('/home/yunkwan/pythonProject/temp/AI_UWB/Data_processed/y_test', y_test)

