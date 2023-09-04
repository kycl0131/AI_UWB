import numpy as np
import pandas as pd
import os
import natsort
import struct


def list_files(directory):
    """
    directory 내의 파일 위치명 리스트를 반환하는 함수
    """
    file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
  
    return file_list

    
def list_files1(train_test_path:str, obj :str ,before_pick):
   
    file_list = list_files(train_test_path)
    # print(f"file_list : {file_list}")
    print(f'folder path : {train_test_path}')
    print(f'object : {obj}')
    # print(file_list)
    # print(train_test_path+"/"+obj)
    file_lists = [x for x in file_list if x.startswith(train_test_path+'/'+obj)]
    
    
    # file_lists = list_files(file_lists[0])
    
    result_stored = []
    
    file_lists = natsort.natsorted(file_lists)
    print(file_lists)
    
    if before_pick == False:
        file_listss = list_files(file_lists[0])
        file_listss = [x for x in file_listss if x.startswith(train_test_path+'/'+obj)]
        file_listss = [x for x in file_listss if x.endswith('.dat')]
        # meta.dat 제외
        file_listss = [x for x in file_listss if not x.endswith('meta.dat')]
        result_stored = result_stored + file_listss
    
    else:
        for i in range(len(file_lists)):
            
            
            file_listss = list_files(file_lists[i])
            # print(f'여기{file_listss}') #test/elecle/elecle~~~
            if "test" in train_test_path:   
                
                file_listss = [x for x in file_listss if x.startswith(train_test_path+'/'+obj)]
                
                file_listss = [x for x in file_listss if x.endswith('.dat')]
                # meta.dat 제외
                file_listss = [x for x in file_listss if not x.endswith('meta.dat')]
                result_stored = result_stored + file_listss
                
            elif "train" in train_test_path:
                file_listss = [x for x in file_listss[i] if x.startswith(train_test_path+'/'+obj)]
                file_listss = [x for x in file_listss if x.endswith('.dat')]
                file_listss = [x for x in file_listss if not x.endswith('meta.dat')]
                result_stored = result_stored + file_listss
        print(result_stored)
  
    return result_stored


def concatData(*args):
    DataClass = [x for lst in args for x in lst]
    return DataClass


def readData(path,things:list,before_pick = False):
    # things = ['elecle','beam','person','yellowmoby'] select one
    count = 0
    print(things[count])
    for ob in things:

        filelist = list_files1(path,ob,before_pick)
        # print(filelist)
        #Labeling
        
        if before_pick == False:
            # Labeling
            label = None
            
            for idx, name in enumerate(things):
                # print(idx,name)
                if ob in name:
                    label = np.full((len(filelist), 1), idx)
                    # print(f'label{label}')
                    # break
            
            if label is None:
                print(f"No matching label found for '{ob}'")

            
            
            # if ob == things[0]:
            #     label = np.full((len(filelist),1 ), 0)
            # elif name in things[1]:
            #     label = np.full((len(filelist),1 ), 1)
            # elif name in things[2]:
            #     label = np.full((len(filelist),1 ), 2)
            # elif name in things[3]:
            #     label = np.full((len(filelist),1 ), 3)
            
            if count == 0:
                labels = label
            else:
                # print(f'label{label.shape}')
                labels = np.concatenate((labels,label),axis=0)
            
    

        count1 = 0
        #Read data
        # print(f'filelist:{(filelist)}')
        
     
        for name in filelist:
            
            print(f'name:{name}')
            
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
            # threshold =0.0005 # max(rawData).round(3).item()  
            # tresh_idx = []  # #(61)- > 1512 ->(1573)-> 1512-> (3085)
            # i = 0
            # while i < len(rawData):
            #     if rawData[i][0] >= threshold:
            #         tresh_idx.append(i)
            #         # print(i)
            #         i += 1340
            #     else:
            #         i += 1
            
            # print(f'threshold:{len(tresh_idx)}')        
          
            if before_pick == True:
                rawData_reshaped = rawData
            else:
              
                rawData_reshaped = rawData.reshape(1,33,-1) 
                
            if count1 ==0:
                OutData = rawData_reshaped
            else:
                OutData = np.concatenate((OutData, rawData_reshaped), axis=0)  
            count1 += 1
        if count == 0:
            OutDatas = OutData
        else:
            OutDatas = np.concatenate((OutDatas,OutData),axis=0)
        print(f'Data shape: {OutDatas.shape}')                    
        count += 1    
            
    
    pt = '/home/yunkwan/project/AI_UWB/Data_processed'
    filename = '_'.join(things)
    if "train" in path:
        if before_pick == True:
            np.save(pt+'/x_unpick_'+str(filename),OutDatas)
        
        elif before_pick == False: 
            np.save(pt+'/y_'+str(filename),labels)
            np.save(pt+'/x_'+str(filename),OutDatas)
    elif "test" in path:
        if before_pick == True:
            np.save(pt+'/x_test_unpick_'+str(filename),OutDatas)
        elif before_pick == False:
            np.save(pt+'/y_test_'+str(filename),labels)
            np.save(pt+'/x_test_'+str(filename),OutDatas)
       
            
    return OutDatas
        
