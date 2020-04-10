#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
import torch
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_12ECG_classifier(data,header_data,classes,model):
    
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class. 
    feats_reshape,feats_external = get_12ECG_features(data,header_data)
    
    feats_reshape = torch.tensor(feats_reshape,dtype=torch.float,device=device)
    feats_external = torch.tensor(feats_external,dtype=torch.float,device=device)
    
    pred = model.forward(feats_reshape,feats_external)
    pred =torch.sigmoid(pred)  
 
    
    tmp_score = pred.squeeze().cpu().detach().numpy()   
    tmp_label = np.where(tmp_score>0.12,1,0)
    for i in range(num_classes):
        if np.sum(tmp_label)==0:
            max_index=np.argmax(tmp_score)
            tmp_label[max_index]=1
        if np.sum(tmp_label)>3:
            sort_index=np.argsort(tmp_score)
            min_index=sort_index[:6]
            tmp_label[min_index]=0
            
            
    
    for i in range(num_classes):
        current_label[i] = np.array(tmp_label[i])
        current_score[i] = np.array(tmp_score[i])

    return current_label, current_score

def load_12ECG_model():
    model = torch.load('resnet_0408.pkl', map_location=device)
    return model
