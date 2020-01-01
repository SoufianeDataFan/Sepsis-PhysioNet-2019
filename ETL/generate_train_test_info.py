#----------------------------------------------------------------------------------------------
'''
What to expect from this script: 

  1- This will generate list of patients with sepsis and no-sepsis in each sets. 
  2- More details be mentioned in the csv files "info_training_train/test"
'''
#----------------------------------------------------------------------------------------------



import os
from multiprocessing import Pool, cpu_count
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

def get_subjects(list_of_examples):
    cols =['subject_id','Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel']  #list(patient.columns[-7:])
    subjects= pd.DataFrame([], columns=cols)
    i=0
    for file in list_of_examples: # tqdm(list_of_examples)
        subject = pd.read_csv(file, sep = "|")
        subject_details = list(subject[cols[1:]].max().values)
        subject_details.insert(0,file.split('.')[0])
        subjects.loc[i]=subject_details
        i+=1
    subjects.rename(columns={'ICULOS':'nb_samples'}, inplace=True)
    return subjects

def generate_info(): 
    for INP_DIR in [TRAIN_DIR, TEST_DIR]:
        if "train" in INP_DIR: 
            prefix= "train"
        else: 
            prefix = 'test'

        os.chdir(INP_DIR)

        list_of_files_train= os.listdir(INP_DIR)
        n= round(len(list_of_files_train)/cpu_count())
        print('n value is ...{}'.format(n))

        files= [list_of_files_train[i:i + n] for i in range(0, len(list_of_files_train), n)]

        with Pool(processes=cpu_count()) as pool:  
            res1 = pool.map(get_subjects, files)

        subjects = pd.concat(res1)

        del res1
        
        # IDs of subjects wih Sepsis 
        Sepsis_subjects_id= subjects.loc[subjects.SepsisLabel.isin([1])].subject_id.values
        # IDs of subjects wihout Sepsis 
        wihoutSepsis_subjects_id= subjects.loc[subjects.SepsisLabel.isin([0])].subject_id.values
        # save files
        output_directory = MAIN_DIR +prefix+'_info'
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        save_file = os.path.join(output_directory, "info_training_"+prefix+".csv")
        subjects.to_csv(save_file, index=False)

        save_file = os.path.join(output_directory, "no_Sepsis_subject_id_"+prefix+".csv")
        pd.DataFrame(wihoutSepsis_subjects_id,
                 columns=['subject_id']).to_csv(save_file, index=False)

        save_file = os.path.join(output_directory, "yes_Sepsis_subject_id_"+prefix+".csv")
        pd.DataFrame(Sepsis_subjects_id, columns=['subject_id']).to_csv(save_file, index=False)
        
        
        print(prefix)
        print("working in this diretory ..... " + INP_DIR)
        
        
        print('generated the following files.....')
        
        print(os.listdir(output_directory))
       

    
    
if __name__ == '__main__':
#----------------------------------------------------------------------------------------------
# how to run the code : 
#---------------- Define MAIN_DIR where you have two files : training_setA and training_setB

    MAIN_DIR = sys.argv[1]
#----------------------------------------------------------------------------------------------
    TRAIN_DIR= MAIN_DIR +'train_data/'
    TEST_DIR = MAIN_DIR +'test_data/'
    train_files = os.listdir(TRAIN_DIR)
    test_files = os.listdir(TEST_DIR)

    # generate files 
    generate_info()
