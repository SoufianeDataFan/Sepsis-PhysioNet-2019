
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.optimizers import RMSprop
from keras import backend as k
from sklearn.preprocessing import normalize


import os
import pandas as pd
import numpy as np
from sys import platform
from IPython.display import display, HTML
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import time
from contextlib import contextmanager
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    

        
if platform == "linux":
    # linux
    INPUT_DIR='C:/Users/soufiane.chami/Desktop/PhysioNet/PhysioNet 2019/training/'
    FILES_DIR= 'C:/Users/soufiane.chami/Desktop/PhysioNet/PhysioNet 2019/'
elif platform == "darwin":
        # OS X
        INPUT_DIR='/Users/macbook/Desktop/PhysioNet/2019/training/'
        FILES_DIR= '/Users/macbook/Desktop/PhysioNet/2019/'
elif platform == "win32":
        # Windows...
        INPUT_DIR='C:/Users/soufiane.chami/Desktop/PhysioNet/PhysioNet 2019/training/'
        FILES_DIR= 'C:/Users/soufiane.chami/Desktop/PhysioNet/PhysioNet 2019/'


def get_subjects():
    cols =['subject_id','Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']  #list(patient.columns[-7:])
    subjects= pd.DataFrame([], columns=cols)
    i=0
    # os.chdir('C:/Users/soufiane.chami/Desktop/PhysioNet 2019/training')
    for file in os.listdir():
        subject = pd.read_csv(file, sep = "|")
        subject_details = list(subject[cols[1:]].max().values)
        subject_details.insert(0,file.split('.')[0])
        subjects.loc[i]=subject_details
        i+=1
    subjects.rename(columns={'ICULOS':'nb_samples'}, inplace=True)
    return subjects

with timer("get genral data about subjects"):
    os.chdir(INPUT_DIR)
    subjects = get_subjects()
    os.chdir(FILES_DIR)
    subjects.to_csv("Subjects_ID_with_Labels.csv", index=False)
    
    # IDs of subjects wih Sepsis 
    Sepsis_subjects_id= subjects.loc[subjects.SepsisLabel.isin([1])].subject_id.values
    pd.DataFrame(Sepsis_subjects_id, columns=['Sepsis_subjects_id']).to_csv("Sepsis_subjects_id.csv", index=False)
    
    # IDs of subjects wihout Sepsis 
    wihoutSepsis_subjects_id= subjects.loc[subjects.SepsisLabel.isin([0])].subject_id.values
    pd.DataFrame(wihoutSepsis_subjects_id,
             columns=['wihoutSepsis_subjects_id']).to_csv("wihoutSepsis_subjects_id.csv", index=False)

    os.chdir(FILES_DIR)
    
with timer("Create Time To Failure columns"):
    list_of_subjects= pd.read_csv('Sepsis_subjects_id.csv')
    df = pd.DataFrame()
    for subject_id in list_of_subjects.Sepsis_subjects_id.values:
        subject= pd.read_csv('training/'+subject_id+'.psv', sep = "|")
        TTF= [f for f in subject.SepsisLabel.values if f==0 ]
        ttf_len= len(TTF)
#         subject = subject.loc[1:ttf_len]
        subject['Subject_ID']= subject_id
        subject['TTF']= [f for f in range(ttf_len-1, -1,-1)] + [0]*(len(subject) - ttf_len)
        df= df.append(subject)

    cols= [ 'Subject_ID', 'TTF','HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2','Age', 'Gender',
           'HospAdmTime', 'ICULOS', 'SepsisLabel']
    data =df[cols].set_index(['Subject_ID', 'TTF']).copy()

    
with timer("handling missing values"):
    y_cols= ["TTF", 'SepsisLabel']
    features = [f for f in data.columns if f not in y_cols+ ['Subject_ID', 'ICULOS']]
    df= data.reset_index().copy()
    df= df[features].interpolate()
    df = df.fillna(0)
    dat1= data.reset_index()[['Subject_ID', 'ICULOS']+y_cols]
    
with timer("Normalize the Data"):
    from sklearn import preprocessing
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, index=data.index, columns=features)
    df = pd.concat([dat1, df.reset_index()[features]], axis=1) 
    
    

 