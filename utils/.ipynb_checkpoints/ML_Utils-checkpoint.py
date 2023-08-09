from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import torch
import joblib

def Dataset_lastVal(data_loader):
    x, y = [], []
    
    for item in data_loader:
        try:
            x = torch.cat((x, item[0][:,-1,:]), dim=0)
            y = torch.cat((y, item[-1]), dim=0)
        except:
            x = item[0][:,-1,:]
            y = item[-1]
            
    return (x, y)


def Dataset_flatten(data_loader):
    x, y = [], []
    
    for item in data_loader:
        try:
            x = torch.cat((x, item[0].reshape(item[0].size()[0], -1)), dim=0)
            y = torch.cat((y, item[-1]), dim=0)
        except:
            x = item[0].reshape(item[0].size()[0], -1)
            y = item[-1]
            
    return (x, y)

def Dataset_3monthAvg(data_loader):
    x, y = [], []
    
    for item in data_loader:
        try:
            x = torch.cat((x, item[0]), dim=0)
            y = torch.cat((y, item[-1]), dim=0)
        except:
            x = item[0]
            y = item[-1]
            
    return (x, y)

def MLOutcome(model, param, weight_name, train_x, train_y, test_x, test_y, status):

    if(status == 'train'):
        GSCV = GridSearchCV(model, param)
        GSCV.fit(train_x, train_y)
        joblib.dump(GSCV, f'{weight_name}.pkl')
    else:
        GSCV = joblib.load(f'{weight_name}.pkl')

    pred_y = GSCV.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    p = precision_score(test_y, pred_y)
    r = recall_score(test_y, pred_y)
    f1 = f1_score(test_y, pred_y, average='macro')

    fpr, tpr, thresholds = metrics.roc_curve(test_y, GSCV.predict_proba(test_x)[:,1], pos_label=1)
    precision, recall, thresholds = metrics.precision_recall_curve(test_y, GSCV.predict_proba(test_x)[:,1], pos_label=1)

    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(recall, precision)


    print('test acc: %.3f | auroc: %.3f | auprc: %.3f | precision: %.3f | recall: %.3f | f1: %.3f' %  (acc, auroc, auprc, p, r, f1))
    # print(f'best parameter: {GSCV.best_params_}')
    # print()
    

