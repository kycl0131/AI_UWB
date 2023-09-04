import argparse
import util.DataOpen as DataOpen 
import util.DataPicker as DataPicker
import numpy as np
import util.CustomTransform as CustomTransform
import matplotlib.pyplot as plt
import logging
import util.TSLoad as TSLoad
import models

from tsai.utils import my_setup
    
#Beam 0 elec 1 person 2 moby 3
def main():

    # "-l is Data Loade and convert to numpy binary file"
    parser = argparse.ArgumentParser(description="A simple command-line argument example")
    parser.add_argument("-l", help="", action="store_true", required=False)
    # parser.add_argument("-show_batch", help="", action="store_true", required=False)
    parser.add_argument("-train", help="", action="store_true", required=False)
    parser.add_argument("-lr", help="", action="store_true", required=False)
    args = parser.parse_args()
    
    # Setup print
    my_setup()
    
    # Data Load and save to numpy
    if args.l:
        print("Preprocessing and Saved in Data_processed")        
        DataOpen.readData('/home/yunkwan/pythonProject/temp/AI_UWB/Data_raw/Record/train',['elecle','beam','person','yellowmoby'])
        DataOpen.readData('/home/yunkwan/pythonProject/temp/AI_UWB/Data_raw/Record/test',['elecle','beam','person','yellowmoby'])

    
    # load binary numpy file 
    path = '/home/yunkwan/pythonProject/temp/AI_UWB/Data_processed'
    x = np.load(path + '/x.npy')
    y = np.load(path + '/y.npy')
    
    

    import struct


    test_data_path = '/home/yunkwan/pythonProject/temp/AI_UWB/230824/230824_elecyle1(마지막에 사람 지나감)/xethru_datafloat_20230824_141824.dat'
    # datafloat file is 4 byte float type 
    with open(test_data_path, 'rb') as file:
        data = file.read()
    # save opened data in rawData list
    rawData = []
    for i in range(0, len(data), 4):  # assuming each float is 4 bytes
        rawData.append(struct.unpack('f', data[i:i+4]))
    #invert to numpy to reshape datasize    
    rawData = np.array(rawData)
    rawData = rawData.reshape(-1,1528)
    rawData.shape

    x_test = DataPicker.pick_data(rawData,x_range=range(200,800))
    y_test = np.full((rawData.shape[0],1 ), 1)
    
    x = DataPicker.remove_bg(x)
    x = x.reshape(-1,33,1528)

    # y = np.concate(y
    
    # x_test = np.load(path + '/x_test.npy')
    # y_test = np.load(path + '/y_test.npy')    
        
    x = x[:,16,:]
    x_test =x_test[:,16,:]
    
    
    print(f'train,val {x.shape}, test {x_test.shape} is loaded')
    
    # Custom Transform
    x = CustomTransform.SetRange(X=x, datasize=1528, num_range_max=1078,start = 100,end =1050, padding =True)
    x_test = CustomTransform.SetRange(X=x_test, datasize= 1528 , num_range_max= 1078 ,start= 100, end= 1050, padding= True)
    
    # Time Series Loader(Split train val)
    # can other transform in TSLoader
    dls = TSLoad.TSLoader(x,y,valid_size= 0.3,stratify =True, shuffle= True,random_state =23,
                          train_batch=128,val_batch=128,num_worker=0)
    

   
    # Model
    MyModel = models.CustomTSModel(dls= dls)
    
    if args.lr:
        MyModel.lrfind()
    
    if args.train:
        MyModel.train(epoch = 100)
        MyModel.validation()
    
    # Test acc
    MyModel.test(x_test,y_test) 
    
    # Visualize Result
    MyModel.result()

    
    # # logging path is  "/home/yunkwaacin/pythonProject/temp/AI_UWB/log"
    # # plot x_test and save
    # logging.basicConfig(filename='/home/yunkwan/pythonProject/temp/AI_UWB/log/log.txt',level=logging.INFO)
    # logging.info(f'x_train shape is {x.shape}')
    # plt.plot(a)
    # plt.savefig('/home/yunkwan/pythonProject/temp/AI_UWB/log/a.png')
    # # print(x_train[0,:,:])

if __name__ == "__main__":
  
    main()
    