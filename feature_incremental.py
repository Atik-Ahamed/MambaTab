from utility import read_data
from config import config
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from mamba_ssm import Mamba
from train_val import train_model

from sklearn.metrics import roc_auc_score
import transtab
import warnings
import numpy as np
from sklearn.utils import class_weight
import sys
from MambaTab import MambaTab

def test_result():
  """
  This function is for inference on the test set.
  """
  model.eval()
  all_test_output_probas=[]

  all_test_labels=[]
  sig=torch.nn.Sigmoid()


  for inputs,labels in dataloader['test']:
    inputs = inputs.unsqueeze(0)
    inputs = inputs.type(torch.FloatTensor)
    inputs = inputs.to(config['device'])
    
    labels = labels.to(config['device'])
    with torch.set_grad_enabled(False):
      outputs = model(inputs)
      outputs=outputs.squeeze()
      outputs=sig(outputs)         
      outputs=outputs.cpu().detach().numpy()
      labels=labels.cpu().detach().numpy()
      for i in range(outputs.shape[0]):
         all_test_labels.append(labels[i])
         all_test_output_probas.append(outputs[i])
  performance_value=roc_auc_score(all_test_labels,all_test_output_probas)
  print("AUROC score: ",performance_value)

class TabularDataLoader(Dataset):
    """
    This is pytorch dataloader. It gives input to the model for train/val/test
    """
    def __init__(self,length,data_type):
        self.length=length
        self.data_type=data_type

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        if self.data_type=='train':
            return x_train[idx],y_train[idx]
        if self.data_type=='val':
            return x_val[idx],y_val[idx]
        if self.data_type=='test':
           return x_test[idx],y_test[idx]
model=MambaTab(input_features=1,n_class=1) # Dummy model initialization. Later this is changed according to feature set length.
for incremental in range(3):
   x_data,y_data=read_data(dataset_name=config['DATASET_NAME'])
   x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=config['SEED'],stratify=y_data, shuffle=True)# Fixed-seed split. so no overlap in iterations
   val_size=int(len(y_data)*0.1)
   x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=val_size,random_state=config['SEED'],stratify=y_train,shuffle=True)
   if incremental!=2:
      subset_size=x_train.shape[1]//3
      x_train=x_train[:,0:subset_size*(incremental+1)]
      x_val=x_val[:,0:subset_size*(incremental+1)]
   print('Subset size:',x_train.shape[1])
            
   train_set = TabularDataLoader(length=x_train.shape[0],data_type='train')
   val_set = TabularDataLoader(length=x_val.shape[0],data_type='val')
   test_set = TabularDataLoader(length=x_test.shape[0],data_type='test')


   dataloader = {
         'train': DataLoader(train_set, batch_size=config['BATCH'], shuffle=True, num_workers=0),
         'val': DataLoader(val_set, batch_size=config['BATCH'], shuffle=False, num_workers=0),
         'test': DataLoader(test_set, batch_size=config['BATCH'], shuffle=False, num_workers=0)
   }
   model.linear_layer=torch.nn.Linear(x_train.shape[1],config['REPRESENTATION_LAYER']) # Here we only change the first layer's input shape for adapting on feature incremental learning setting
   model=model.to(config['device'])
   model=train_model(model,config, dataloader)
      
test_result()
print("----------------Complete----------------")


