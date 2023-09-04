import sklearn.metrics as skm 
from tsai.all import build_ts_model, accuracy ,Learner
from tsai.models import RNN_FCN
import matplotlib.pyplot as plt

class CustomTSModel:    
    def __init__(self,dls):

        self.dls = dls
        MLSTM_FCN = RNN_FCN.MLSTM_FCN
        self.model = build_ts_model(MLSTM_FCN, dls =dls) #MLSTM_FCN
        self.learn = Learner(self.dls, self.model, metrics=accuracy)
        self.learn.save('stage0')
        self.learn.load('stage0')
        self.plt = plt
        # print('init')
    
    def lrfind(self):
        self.learn.lr_find()
        self.plt.savefig('/home/yunkwan/pythonProject/temp/AI_UWB/log/lr_find.png')
        self.plt.close()


    def train(self,epoch =100):
      
        print("Train")
        from tsai.all import ts_learner,ShowGraph
        from fastai.callback.schedule import fit_one_cycle

        # learn = ts_learner(dls,metrics=accuracy, cbs = ShowGraph())
        self.learn.fit_one_cycle(epoch, lr_max=1e-3)
        self.learn.save('stage1')
        
        self.learn.recorder.plot_metrics()

        self.learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner') 
        self.plt.savefig('/home/yunkwan/project/AI_UWB/log/train.png')
        self.plt.close()
        # del learn, dsets, dls
        
    def validation(self):
        from tsai.all import load_learner_all
        self.learn = load_learner_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
        self.dls = self.learn.dls
        self.valid_dl = self.dls.valid
        valid_probas, valid_targets, valid_preds = self.learn.get_preds(dl=self.valid_dl, with_decoded=True)
        print(f'validation accuracy: {(valid_targets == valid_preds).float().mean()}')
        
    def test(self,x_test,y_test ):
    
        # x_test = to3d(x_test)
        
        y_test = y_test.squeeze()
        self.valid_dl = self.dls.valid
        self.test_ds = self.valid_dl.dataset.add_test(x_test,y_test)# In this case I'll use X and y, but this would be your test data
        self.test_dl = self.valid_dl.new(self.test_ds)
        print(next(iter(self.test_dl)))
        print('여기까지')
        test_probas, test_targets, test_preds = self.learn.get_preds(dl=self.test_dl, with_decoded=True, save_preds=None, save_targs=None)
        
        print(f'test accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}') 
        print(f'test_probas: {test_probas},test_targets: {test_targets},test_preds: {test_preds}')

    def result(self):
        # from tasi.all import Classificationinterpretation
        from fastai.interpret import ClassificationInterpretation

        
        self.learn.show_probas()
        self.plt.savefig('/home/yunkwan/project/AI_UWB/log/probas.png')
        self.plt.close()
        
        interp = ClassificationInterpretation.from_learner(self.learn,ds_idx=0)
        interp.plot_confusion_matrix()
        self.plt.savefig('/home/yunkwan/project/AI_UWB/log/confusion_matrix_train.png')
        self.plt.close()
        
        
        interp = ClassificationInterpretation.from_learner(self.learn,ds_idx=1)
        interp.plot_confusion_matrix()
        self.plt.savefig('/home/yunkwan/project/AI_UWB/log/confusion_matrix_val.png')
        self.plt.close()
        
    
        # self.dls.show_batch(sharey=True, figsize=(18,6),max_n=80,ncols=8,)
        # self.plt.savefig('/home/yunkwan/pythonProject/temp/AI_UWB/log/train_batch.png')
        # self.plt.close()
        
        self.learn.show_results(sharey=True,ds_idx=0, max_n=80,ncols=3,figsize=(18,6))
        self.plt.savefig('/home/yunkwan/project/AI_UWB/log/train_result.png')
        self.plt.close()
        
        self.learn.show_results(sharey=True, dl=self.test_dl,max_n=80,ncols=3,figsize=(18,6))
        self.plt.savefig('/home/yunkwan/project/AI_UWB/log/test_result.png')