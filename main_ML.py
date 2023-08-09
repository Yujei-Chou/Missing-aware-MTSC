from utils.Normalize import *
from utils.ML_Utils import *
import argparse
import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# model define
parser = argparse.ArgumentParser()

parser.add_argument('--seq_len', type=int, default=12, help='sequence length')
parser.add_argument('--max_depth', type=int, default=15, help='RF and DT max depth')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--norm', type=str, default='Normalize', help='Standarlize / Normalize')
parser.add_argument('--horizon', type=int, default=1, help='forcast horizon')
parser.add_argument('--drop_ratio', type=float, default=0.3, help='drop ratio')

parser.add_argument('--model', type=str, default='DT', help='DT / RF / XGB')
parser.add_argument('--format', type=str, default='DT', help='LastTS / Last3TS / Flatten')
parser.add_argument('--testing_type', type=str, default='balance', help='balance / imbalance / custom')
parser.add_argument('--testing_path', type=str, default='dataset/preprocessed_data/Experts_Questions/Sample100_form1', help='path for the pretrained weights')
parser.add_argument('--weight_path', type=str, default='weight/ML/DT_LastMonth', help='path for saving the weights')
parser.add_argument('--status', type=str, default='train', help='train / test')
args = parser.parse_args()


# Setting new dataset and dataloader
from data_provider.dataset_ml import *
from torch.utils.data import DataLoader

with open(f'dataset/preprocessed_data/training_balance_{args.horizon}yearDrop{int(100*args.drop_ratio)}%.pickle','rb') as f:
    train_data = pickle.load(f)

with open(f'dataset/preprocessed_data/validation_balance_{args.horizon}yearDrop{int(100*args.drop_ratio)}%.pickle','rb') as f:
    validate_data = pickle.load(f)

if(args.testing_type == 'custom'):
    with open(f'{args.testing_path}.pickle','rb') as f:
        test_data = pickle.load(f)
else:
    with open(f'dataset/preprocessed_data/testing_{args.testing_type}_{args.horizon}yearDrop{int(100*args.drop_ratio)}%.pickle','rb') as f:
        test_data = pickle.load(f)

    
base = train_data[0]
train_lab_norm, valid_lab_norm, test_lab_norm = lab_Norm(args.norm, args.seq_len , base, train_data[0], validate_data[0], test_data[0])  # Normalize the laboratory data
train_patient_norm, valid_patient_norm, test_patient_norm = patient_Norm(args.norm, train_data[1], validate_data[1], test_data[1])  # Normalize the patient data

train_label, valid_label, test_label = list(zip(train_data[3], train_data[4], train_data[2])), list(zip(validate_data[3], validate_data[4], validate_data[2])), list(zip(test_data[3], test_data[4], test_data[2]))
train_dataset, validate_dataset, test_dataset = Dataset(list(zip(train_lab_norm, train_patient_norm)), train_label), Dataset(list(zip(valid_lab_norm, valid_patient_norm)), valid_label), Dataset(list(zip(test_lab_norm, test_patient_norm)), test_label)


print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda batch: train_dataset.collate_fn(batch, args.format), shuffle=True)
validate_data_loader = DataLoader(validate_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda batch: validate_dataset.collate_fn(batch, args.format), shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda batch: test_dataset.collate_fn(batch, args.format), shuffle=False)

print(f'number of training data: {len(train_data[0])}')
print(f'number of validation data: {len(validate_data[0])}')
print(f'number of testing data: {len(test_data[0])}')
print()

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


if(args.format == 'LastTS'):     # Last 1 Time Step Average
    train_x, train_y = Dataset_lastVal(train_data_loader)
    test_x, test_y = Dataset_lastVal(test_data_loader)
elif(args.format == 'Last3TS'):  # Last 3 Time Step Average
    train_x, train_y = Dataset_3monthAvg(train_data_loader)
    test_x, test_y = Dataset_3monthAvg(test_data_loader)
else:                            # All Time Step Flatten
    train_x, train_y = Dataset_flatten(train_data_loader)
    test_x, test_y = Dataset_flatten(test_data_loader)


if(args.model == 'RF'):
    model = RandomForestClassifier(random_state=seed)
    param = {'n_estimators':[10,50,100,150,200],'max_depth':[5,10,15,20]}
elif(args.model == 'DT'):
    model = DecisionTreeClassifier(random_state=seed)
    param = {'max_depth':[5,10,15,20]} 
else:
    model = XGBClassifier(random_state=seed)
    param = {'n_estimators':[10,50,100,150,200],'max_depth':[5,10,15,20]}   


MLOutcome(model, param, args.weight_path, train_x, train_y, test_x, test_y, args.status)
