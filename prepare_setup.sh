echo "Download the Data ... "

wget https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip
wget https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip
mkdir data/
unzip -qq training_setA.zip
unzip -qq training_setB.zip
mv training/* data/
mv training_setB/* data/

rm -rf *.zip
rm -rf training/
rm -rf training_setB/

echo "It's all done .. "

echo "Total number of file in data/ is ..."

ls data/ | wc -l

python ~/sepsis_challenge_2019/ETL/split_data.py

python ~/sepsis_challenge_2019/ETL/generate_train_test_info.py ~/sepsis_challenge_2019/
