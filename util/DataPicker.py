
import matplotlib.pyplot as plt
import numpy as np
import struct


def remove_bg(rawData):
    if rawData.ndim ==3:
        processed_data = rawData[:,:,:].copy()
        processed_data = processed_data.reshape(33*processed_data.shape[0],1528) 
    # lim = 0.004
    # plt.ylim(-lim, lim)
    # x_range = x_range
    else:
        rawData = rawData.reshape(-1,1528)
        processed_data = rawData[:,:].copy()
    
    

    
    diff_data = np.diff(processed_data, axis=0)
    processed_data[1:, :] = diff_data
    processed_data[0, :] = 0
    print(f'removed background shape{processed_data.shape}')
    
    return processed_data


def pick_data(test_data_path,start ,end ,threshold = 0.00015,datasize=1528, num_range_max=1078):
    
    #set Range
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
    
    x_range =range(start,end)
    
    if test_data_path.ndim ==3:
        print(test_data_path.shape)
        test_data_path = test_data_path.reshape(33*test_data_path.shape[0],len(x_range))        
    else:
        test_data_path = test_data_path.reshape(-1,1528)
    # lim = 0.004
    # plt.ylim(-lim,lim)
    
    rawData = test_data_path.copy()
    
    #Simple Remove BG
    rawData_bgx = remove_bg(rawData)

    print(rawData_bgx.shape)
    

    
    
    
        
    # rawData_bgx = rawData
    # see only this range
    
    # th is threshold (hyper parameter)
    
    th = np.full(len(x_range),threshold)
    print(th.shape)
    # Save the index of the data that is above the threshold
    # count = [] 
    # for i in range(0,rawData_bgx.shape[0]):    
    #     Data = rawData_bgx[i,x_range]#-rawData_bgx[0,x_range]
        
    #     if np.any((Data-th)>0):
    #         # Data = Data-th
    #         plt.plot(Data)
    #         # print(i)
    #         count.append(i)

    count = np.any(rawData_bgx[:,x_range] - th > 0, axis=1)
    indices = np.where(count)[0]
    # print(f'indices : {indices}')
    
    # plt.plot(rawData_bgx[indices][:, :].T)




    # plt.show()
    # plt.close()
    # print((indices))
        

    # 비슷한 숫자를 모아서 각 배열에 넣는 작업 (임계값 이상 )
    result = []
    current_group = []
    # 33 * 1528 에서 33//2 = 16.5 대략 겹치는 범위 17
    for num in indices:
        if not current_group or abs(current_group[-1] - num) <= 17:
            current_group.append(num)
        else:
            if len(current_group) >= 5:
                result.append(current_group)
            current_group = [num]

    if current_group:
        if len(current_group) >= 5:
            # print(f'current_group : {current_group}')
            result.append(current_group)

    # print(result)
    group_idx = []
    for group in result:
        group_idx_temp = []
        cent_idx = len(group) // 2
        
        # 현재 그룹의 숫자들을 중심을 기준으로 앞뒤 16개의 숫자를 계산하여 리스트에 추가
        for offset in range(-16, 17):
            idx = group[cent_idx] + offset
            group_idx_temp.append(idx)
        # print(len(group_idx_temp))
        group_idx.append(group_idx_temp)

    
    #stack for empty numpy array
    result = np.empty((len(group_idx),33,1528), dtype=float)
    # print(group_idx)
    # print(f'group_idx : {len(group_idx)}')
    # print(f'result : {result.shape}')
    #stack rawData
    for n in range(len(group_idx)):
        count = 0
        for i in group_idx[n]:
            if i == group_idx[n][0]:
                k = i -1
                
            result[n,count,:] = rawData[i] - rawData[k]
            # print( i,k)
            count += 1


    return result


def pick_data_unremoved(test_data_path,start ,end ,threshold = 0.00015,datasize=1528, num_range_max=1078):
    
    #set Range
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
    
    x_range =range(start,end)
    
    if test_data_path.ndim ==3:
        print(test_data_path.shape)
        test_data_path = test_data_path.reshape(33*test_data_path.shape[0],len(x_range))        
    else:
        test_data_path = test_data_path.reshape(-1,1528)
    # lim = 0.004
    # plt.ylim(-lim,lim)
    
    rawData = test_data_path.copy()
    
    #Simple Remove BG
    rawData_bgx = remove_bg(rawData)

    print(rawData_bgx.shape)
    

    
    
    
        
    # rawData_bgx = rawData
    # see only this range
    
    # th is threshold (hyper parameter)
    
    th = np.full(len(x_range),threshold)
    print(th.shape)
    # Save the index of the data that is above the threshold
    # count = [] 
    # for i in range(0,rawData_bgx.shape[0]):    
    #     Data = rawData_bgx[i,x_range]#-rawData_bgx[0,x_range]
        
    #     if np.any((Data-th)>0):
    #         # Data = Data-th
    #         plt.plot(Data)
    #         # print(i)
    #         count.append(i)

    count = np.any(rawData_bgx[:,x_range] - th > 0, axis=1)
    indices = np.where(count)[0]
    # print(f'indices : {indices}')
    
    # plt.plot(rawData_bgx[indices][:, :].T)




    # plt.show()
    # plt.close()
    # print((indices))
        

    # 비슷한 숫자를 모아서 각 배열에 넣는 작업 (임계값 이상 )
    result = []
    current_group = []
    # 33 * 1528 에서 33//2 = 16.5 대략 겹치는 범위 17
    for num in indices:
        if not current_group or abs(current_group[-1] - num) <= 17:
            current_group.append(num)
        else:
            if len(current_group) >= 5:
                result.append(current_group)
            current_group = [num]

    if current_group:
        if len(current_group) >= 5:
            # print(f'current_group : {current_group}')
            result.append(current_group)

    # print(result)
    group_idx = []
    for group in result:
        group_idx_temp = []
        cent_idx = len(group) // 2
        
        # 현재 그룹의 숫자들을 중심을 기준으로 앞뒤 16개의 숫자를 계산하여 리스트에 추가
        for offset in range(-16, 17):
            idx = group[cent_idx] + offset
            group_idx_temp.append(idx)
        # print(len(group_idx_temp))
        group_idx.append(group_idx_temp)

    
    #stack for empty numpy array
    result = np.empty((len(group_idx),33,1528), dtype=float)
    # print(group_idx)
    # print(f'group_idx : {len(group_idx)}')
    # print(f'result : {result.shape}')
    #stack rawData
    for n in range(len(group_idx)):
        count = 0
        for i in group_idx[n]:
            if i == group_idx[n][0]:
                k = i -1
                
            result[n,count,:] = rawData[i]
            # print( i,k)
            count += 1


    return result

