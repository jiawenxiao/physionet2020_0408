#!/usr/bin/env python

import numpy as np

def get_12ECG_features(data,header_data):
    set_length=10000
    resample_interval=2
    data_num=np.zeros((1,12,set_length))

    length=data.shape[1]
    if length>=set_length:
        data_num[:,:,:]=data[:,:set_length]
    else:
        data_num[:,:,:length]=data
       
    
    resample_index=np.arange(0,set_length,resample_interval).tolist()
    
    data_num=data_num[:,:,resample_index]  
    
    data_external=np.zeros((1,3))
    
    for lines in header_data:
        if lines.startswith('#Age'):
            age=lines.split(': ')[1].strip()
            if age=='NaN':
                age='60'     
        if lines.startswith('#Sex'):
            sex=lines.split(': ')[1].strip()
            
            
    length=data.shape[1]
    data_external[:,0]=float(age)/100
    data_external[:,1]=np.array(sex=='Male').astype(int) 
    data_external[:,2]=length/30000   
    
    return data_num,data_external