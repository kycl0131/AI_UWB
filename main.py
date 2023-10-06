import argparse
import util.DataOpen as DataOpen 
import util.DataPicker as DataPicker
import numpy as np
import util.CustomTransform as CustomTransform
import matplotlib.pyplot as plt
import logging
import util.TSLoad as TSLoad
from tsai.utils import my_setup
import models
    
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
        print("Preprocessing and Saved in Data_processed")        # ['elecle','beam','person','yellowmoby']
        DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_raw/Record/train',['elecle','person'],before_pick= False)
        DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_raw/Record/test',['elecle','person'],before_pick= False)


    #------------------------------ temp------------------------------------------------------------------------
    from sklearn.preprocessing import MinMaxScaler
    
    # elec = DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_new/test 임시',['elecle'],before_pick= True)
    # elec = DataPicker.pick_data(elec,start=50,end= 600)
    # beam = DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_new/test 임시',['beam'],before_pick= True)
    # beam = DataPicker.pick_data(beam,start=50,end= 600)
    # X = np.concatenate((elec,beam),axis=0)
    # X = CustomTransform.SetRange(X=X, datasize=1528, num_range_max=1078,start = 50,end =600, padding =True,ndim=3)
    # shape = X.shape[2]
    # elec_y =np.full(elec.shape[0],1)
    # beam_y =np.full(beam.shape[0],2)
    # y = np.concatenate((elec_y,beam_y),axis=0)
    # print(f'X shape is {X.shape}')
    # print(f'y shape is {y.shape}')
    X = np.load('/home/yunkwan/project/AI_UWB/Data_processed/x_elecle_beam_person_yellowmoby.npy')
    X = CustomTransform.SetRange(X=X, datasize=1528, num_range_max=1078,start = 0,end =1528, padding =True,ndim=3)
    shape = X.shape[2]
    y = np.load('/home/yunkwan/project/AI_UWB/Data_processed/y_elecle_beam_person_yellowmoby.npy')
    from sklearn.model_selection import train_test_split
    X_gan, X_cl, y_gan, y_cl = train_test_split(X, y, test_size=0.5, random_state=42)
    X_cl,X_cltest, y_cl, y_cltest = train_test_split(X_cl, y_cl, test_size=0.3, random_state=42)
    X_cl = X_cl.reshape(-1,shape)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X1 = scaler.fit_transform(X_cl)
    y1 = np.repeat(y_cl, 33)
    y1 = y1.reshape(-1,1)

    dls = TSLoad.TSLoader(X1,y1,valid_size= 0.3,stratify =True, shuffle= True,random_state =23,
                          train_batch=128,val_batch=128,num_worker=0)
    
 
    
    
    #---------------------------------------------------------------------------------------------------------
    # load binary numpy file 
    
    # path = '/home/yunkwan/project/AI_UWB/Data_processed'
    # x = np.load(path+'/x_elecle_beam_person_yellowmoby.npy')
    # y = np.load(path + '/y_elecle_beam_person_yellowmoby.npy')
  
    # x_test = np.load(path + '/x_test_unpick_person.npy')
    # y_test = np.load(path + '/y_test_unpick_.npy')    /
    
    # elec = DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_new/test 임시',['elecle'],before_pick= True)
    # elec = np.load('/home/yunkwan/project/AI_UWB/Data_processedtemp_elec.npy')
    # beam = DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_new/test 임시',['beam'],before_pick= True)
    # person = DataOpen.readData('/home/yunkwan/project/AI_UWB/Data_new/test',['person'],before_pick= True)   
    
    
    # x_test = np.load(path + '/x_test_unpick.npy')
    # elec = DataPicker.pick_data(elec,start=150,end= 600)
    # person = DataPicker.pick_data(person,start=150,end= 600)
    # x_test=elec
    # x_test = np.concatenate((elec,person),axis=0)
    # print(f'x_test shape is {x_test.shape}')
    # elec_y =np.full(elec.shape[0],1)
    # person_y =np.full(person.shape[0],2)
    # y_test = elec_y
    # y_test = np.concatenate((elec_y,person_y),axis=0)
 
    # Custom Transform
    # x = CustomTransform.SetRange(X=x, datasize=1528, num_range_max=1078,start = 150,end =600, padding =True,ndim=3)
    # x_test = CustomTransform.SetRange(X=x_test, datasize= 1528 , num_range_max= 1078 ,start= 150, end= 600, padding= True,ndim=3)
    
 
    # print(f'train,val {x.shape, y.shape } is loaded')
    # print(f'test {x_test.shape, y_test.shape} is')
    
    # Time Series Loader(Split train val)
    # can other transform in TSLoader
    # dls = TSLoad.TSLoader(x,y,valid_size= 0.3,stratify =True, shuffle= True,random_state =23,
    #                       train_batch=128,val_batch=128,num_worker=0)
    
    # dls = TSLoad.TSLoader(x_test,y_test,valid_size= 0.5,stratify =True, shuffle= True,random_state =23,
    #                       train_batch=8,val_batch=8,num_worker=0)
    
   
    # Model
    MyModel = models.CustomTSModel(dls= dls)
    # MyModel.predict(x_test)
    if args.lr:
        MyModel.lrfind()
    
    if args.train:
        MyModel.train(epoch = 1)
        # MyModel.validation()
    
    # Test acc
    # MyModel.validation()
    # MyModel.test(X_cltest,y_cltest) 
    
    # Visualize Result
    # MyModel.result()

    
    # # logging path is  "/home/yunkwaacin/pythonProject/temp/AI_UWB/log"
    # # plot x_test and save
    # logging.basicConfig(filename='/home/yunkwan/pythonProject/temp/AI_UWB/log/log.txt',level=logging.INFO)
    # logging.info(f'x_train shape is {x.shape}')
    # plt.plot(a)
    # plt.savefig('/home/yunkwan/pythonProject/temp/AI_UWB/log/a.png')
    # # print(x_train[0,:,:])
    #
if __name__ == "__main__":
  
    main()
    