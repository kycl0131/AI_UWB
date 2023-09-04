def TSLoader(x,y, valid_size =0.3,stratify = False , shuffle = False,random_state =23, train_batch =128,val_batch =128,num_worker =0):    
    from tsai.all import to3d, get_splits,plot_splits
    from tsai.all import TSDatasets,TSDataLoaders,Categorize,TSStandardize
    import sklearn.metrics as skm
    import numpy as np
    import logging
    import matplotlib.pyplot as plt
    
    
    x = x
    y = y
    
    X_3d = to3d(x)
    y = y.squeeze()

    # To logging splits , change get_splits -> plot_splits -> add plt.savefig
    splits = get_splits(y.squeeze(), valid_size=valid_size, stratify=stratify, random_state=random_state, shuffle=shuffle )
    print('DataLoading-----------------------------------------------')
    plt.savefig('/home/yunkwan/project/AI_UWB/log/split.png')
    plt.close()
    
    train_idx , valid_idx = splits
    valid_idx   

    tfms  = [None, [Categorize()]]
    dsets = TSDatasets(X_3d, y, tfms= tfms, splits=[train_idx,valid_idx], inplace=True)
    dsets

    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[train_batch, val_batch], batch_tfms=[TSStandardize()], num_workers=num_worker)

    return dls