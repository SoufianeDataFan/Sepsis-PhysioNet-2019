import os
import pandas as pd
import numpy as np
import sys
from sys import platform
from IPython.display import display, HTML
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import time
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sys import platform


WDIR = os.path.dirname(os.path.abspath(__file__))

MAIN_DIR = '/'.join(WDIR.split("/")[:-1])

DATA_DIR =  MAIN_DIR + '/data/' 

# you may face a problem above . You edit it accodring to your directory 
# you can fix by adding directly the DIR of /data/ file like my case:  MAIN_DIR = '/home/chami_soufiane_fr/PhysioNet2019/data/'


data_files = os.listdir(DATA_DIR)
#---------------------------------------------------------------------------------------------

def get_subjects(list_of_examples):
    cols =['subject_id','Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']  #list(patient.columns[-7:])
    subjects= pd.DataFrame([], columns=cols)
    i=0
    for file in tqdm(list_of_examples):
        subject = pd.read_csv(file, sep = "|")
        subject_details = list(subject[cols[1:]].max().values)
        subject_details.insert(0,file.split('.')[0].split('/')[-1])
        subjects.loc[i]=subject_details
        i+=1
    subjects.rename(columns={'ICULOS':'nb_samples'}, inplace=True)
    return subjects


from multiprocessing import Pool, cpu_count
n = round(len(data_files)/cpu_count())+1
list_of_files_train= [ DATA_DIR + f for f in data_files]
files= [list_of_files_train[i:i + n] for i in range(0, len(list_of_files_train), n)]

with Pool(processes=cpu_count()) as pool:  
    print('processing ..')
    res1 = pool.map(get_subjects, files)
    
pool.close() # shut down the pool

# concatenate the results from all cores 
data = pd.concat(res1)



n1= data[data.SepsisLabel==0].shape[0]
n2= data[data.SepsisLabel==1].shape[0]
print('number of sepsis patients is .....{}'.format(n1))
print('number of NON-sepsis patients is .....{}'.format(n2))

print('ratio of sepsis patients in the data is .....{}'.format(round(n2/n1*100)) + '%')


from sklearn.model_selection import train_test_split
X_train, X_test= train_test_split(data,
                                  random_state=88,
                                  stratify=data.SepsisLabel, 
                                  test_size=0.25)

list_train_files = X_train.subject_id.values 
list_test_files = X_test.subject_id.values 

data[data.subject_id.isin(list_train_files)].to_csv('train_data_details.csv', index=False)
data[data.subject_id.isin(list_test_files)].to_csv('test_data_details.csv', index=False)
data.to_csv('all_data_details.csv', index=False)

print('Files are saved in :....... '+ os.getcwd())





# move data to train and test dir 

os.mkdir(MAIN_DIR + '/train_data/')
os.mkdir(MAIN_DIR + '/test_data/')


import os
from os import path
import shutil

src = DATA_DIR 

def move_files(list_f): 
    for f in list_f:
        shutil.copy(f, dst)
        
for status in ['train', 'test']: 
    print(status)
    dst = MAIN_DIR + '/train_data/'
    list_of_files= list(DATA_DIR+ list_train_files + '.psv')

    if status=='test': 
        list_of_files= list(DATA_DIR+ list_test_files + '.psv')
        dst = MAIN_DIR + '/test_data/'
    
    from multiprocessing import Pool, cpu_count
    n = round(len(list_of_files)/cpu_count())+1
    files= [list_of_files[i:i + n] for i in range(0, len(list_of_files), n)]

    with Pool(processes=cpu_count()) as pool:  
        print('processing ..')
        res1 = pool.map(move_files, files)

    pool.close() # shut down the pool

