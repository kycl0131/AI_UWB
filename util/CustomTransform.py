import numpy as np

def SetRange(X,datasize,num_range_max,start,end, padding =None, ndim=3):
    print(f'SetRange{start}~{end}---------------------------------------------')

    # datasize = datasize
    # num_range_max = num_range_max
    # start = start
    # end =end
    
    # 1078
    rangevec = np.arange(0,datasize,datasize/num_range_max) # detect range 0~10m -> 0~ 1000
   
    # rangevec = np.arange(0,num_range_max,num_range_max/datasize)
    # start_id= np.where(np.floor(rangevec) == start)
    start_id = np.abs(rangevec - (datasize/num_range_max)*start)
    start_id = np.argmin(start_id)
    # 가장 가까운 값의 인덱스를 찾기 위해 절댓값 차이 계산
   
    diff_end = np.abs(rangevec - (datasize/num_range_max)*end )
    # print(f'ddif{diff_end}')
    # 가장 작은 차이를 갖는 인덱스 반환
    closest_index = np.argmin(diff_end)
    # start_id == cl
    # end_id = np.where(np.round(rangevec) == end)
    # print(f'cl:{closest_index }')
    start = start_id
    # end = end_id[0][0]
    end = closest_index

    if ndim ==3:
        x = X[:,:,start:end]#.reshape(-1,33,end-start)
           # zero padding 1d
        if padding is True:
            x = np.pad(x, ((0,0),(0,0),(start, num_range_max - end)), 'constant', constant_values=0)        
        # print(x.shape)   
    elif ndim ==2:
        x = X[:,start:end]#.reshape(-1,end-start)
        if padding is True:
            x = np.pad(x, ((0,0),(start, num_range_max - end)), 'constant', constant_values=0)        
        # print(x.shape)
 
    return x
    
    