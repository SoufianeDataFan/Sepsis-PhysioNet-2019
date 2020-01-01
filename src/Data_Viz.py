'''
require as inputs :
    - features_improtance
    - oof_preds_
    - y_ture
    - folds_idx_
'''



import seaborn as sns
from matplotlib import pyplot as plt
def display_importances(feature_importances, score):
    cols = feature_importances.sort_values(by='Importance', ascending=False)['feature'].values
    best_features =feature_importances.sort_values(by='Importance', ascending=False).head(50)
    # best_features.reset_index(inplace=True)
    plt.figure(figsize=(15, 10))
    sns.barplot(x="Importance", y="feature", data=best_features)
    plt.title('LightGBM features (avg over folds)')
    plt.tight_layout()

    import datetime
    today= datetime.datetime.now()
    featImp_file_name ='features_importance'+today.strftime('__%H:%M:%S_%d_%b_%Y_')+'_score_'+str(float("%0.4f"%score))+'.png'
    plt.savefig(featImp_file_name)



def display_precision_recall(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))

    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig('recall_precision_curve.png')

def display_roc_curve(y_, oof_preds_, folds_idx_):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        # Plot the roc curve
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig('roc_curve.png')


import datetime
def make_submission_dir(score, undersampling=False):
    today= datetime.datetime.now()
    # mkdir
    prefix='iteration'+today.strftime('__%H:%M:%S_%d_%b_%Y_')+"Score__" +str(score)
    prefix= 'undersampling_'+prefix if undersampling else prefix
    new_dir= WDR+'/'+prefix
    os.mkdir(new_dir)
    os.chdir(new_dir)
    print('done')
    os.system('sudo cp /home/chami_soufiane_fr/feature-engineering-lightgbm.ipynb '+new_dir+'/script.ipynb')
    return prefix


prefix= make_submission_dir(score)



# Display a few graphs
# from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score

# display_importances(feature_importances, score)

# folds = StratifiedKFold(n_splits=5,random_state=820)

# folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(X, y)]

# display_roc_curve(y_= y, oof_preds_=y_oof, folds_idx_=folds_idx)

# display_precision_recall(y_=y, oof_preds_=y_oof, folds_idx_=folds_idx)


# submission_file = 'submission_'+prefix+'.csv'
# sub.to_csv(submission_file, index=False)
# print("submission is made in this directory : " + os.getcwd() + "\n")
# print("submission file_name : " + submission_file+ "\n")

# Message='"normal 5 stratified folds + twisted parameters"'
# Command = 'kaggle competitions submit ieee-fraud-detection -f '+submission_file+" -m "+ Message
# os.system(Command)
