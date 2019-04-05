# Early Prediction of Sepsis in ICU- An Efficient Feature Selection and Classification Approach



## Space for Authors Name and Affiliations


**Objective** : Sepsis is a severe medical condition caused by body’s extreme response to an infection causing tissue damage, organ failure and even death. It is a global health concern whose early diagnosis is most. Our objective is to develop a machine learning model to predict sepsis at least 6 hours before clinical recognition from the available physiological data provided by PhysioNet/Computing in Cardiology (CinC) challenge 2019.

**Approach**: Data is collected in hourly basis among 40, 336 patients (Males, Females) from Intensive Care Unit (ICU) in two separate hospitals. In total, 40 time dependent features of vital signs, laboratory values and demographics are acquired from each patient. The dataset is divided into 80% training and 20% testing. Test data is divided into 3, 6, 9 and 12 hours slices for each patient. A recurrent neural network (add other classifier) is used in a nested cross-validation architecture for classification. Features are selected sequentially in inner loop using 5-fold cross validation and the selected features are used in the outer loop via patient-wise external CV for classification. The prediction of sepsis is based on the obtained probability of detection at 50% threshold in hourly basis on the test data.  

**Main Results**: We obtained the optimal accuracy of (value), sensitivity of (value), and specificity of (value) from our proposed algorithm within the early detection timeframe of 6 hours. Our model achieved the overall score of (mean±sd)% from the utility function assessed by PhsioNet. These are the preliminary results and further areas of improvement have been identified. 

**Significance**: Sepsis can be predicted at least 6 hours in advance using only (number) features. This provides early treatment thereby, reducing length of stay in hospitals, cost of medical expenses and mortality rate. 
Keywords: sepsis, early prediction, classifier name


