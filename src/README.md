# Important note:

1. The complete Code of my paper will be public after the publication process of the CinC Conference is finished.



# How to does it work?

Go to desktop and run :

1. `git clone https://github.com/SoufianeDataFan/sepsis_challenge_2019.git; cd sepsis_challenge_2019/`

3. `sh prepare_setup.sh`

You will have now the



# Get data


First, the whole training data is `training_setA.zip` and `training_setB.zip`

run `sh prepare_setup.sh` to download the data and get it in one directory.



# Test setup

In order to evaluate any training model, we'll create a unified validation setup.

We'll train the succesful model on the data for submission.

For this purpose, we run `~/sepsis_challenge_2019/etl/split_data.py` inside `prepare_setup.sh`



At the end of compiling `prepare_setup.sh`

You have three files saved : `train_data_details.csv` , `test_data_details.csv` containing the patient IDs that will be used in the trainset and testset.


Last Update: July, 11th 2019
