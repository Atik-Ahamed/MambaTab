import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler,MinMaxScaler
import torch
def read_data(dataset_name):
    data=pd.read_csv('datasets/'+dataset_name+'/data_processed'+'.csv')
    
    #fill nulll values
    for col in data.columns: 
        data[col].fillna(data[col].mode()[0], inplace=True)

    #categorical encoder
    for c in data.columns:
        if is_string_dtype(data[c]):
            data[c]=data[c].str.lower()
            enc=OrdinalEncoder()
            cur_data=np.array(data[c])
            cur_data=np.reshape(cur_data,(cur_data.shape[0],1))
            data[c] = enc.fit_transform(cur_data)

    y_data=data[data.columns[-1]]
    x_data = data.drop(labels = [data.columns[-1]],axis = 1)
    x_data=MinMaxScaler().fit_transform(x_data)
    x_data,y_data=np.array(x_data),np.array(y_data)
    return x_data,y_data