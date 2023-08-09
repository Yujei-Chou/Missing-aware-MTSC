import argparse
from utils.CKD_EPI import CKD_EPI
from utils.GetPastFuture import getPastList, getFutureList
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# Define arguments
parser = argparse.ArgumentParser()

parser.add_argument('--lab_data', default='dataset/lab_data.csv', help='Path for lab data')
parser.add_argument('--patient_data', default='dataset/patient_data.csv', help='Path for patient data')
parser.add_argument('--MPerTS', type=int, default=1, help='Number of months per time step')
parser.add_argument('--PastTSs', type=int, default=12, help='Time span in the past')
parser.add_argument('--FutureTSs', type=int, default=12, help='Time span in the future')
parser.add_argument('--drop_ratio', type=float, default=0.3, help='eGFR drop ratio')
parser.add_argument('--cut1_PID', type=int, default=12806915, help='Pick patient ID for 1st cut')
parser.add_argument('--cut2_PID', type=int, default=14770140, help='Pick patient ID for 2nd cut')
parser.add_argument('--train_data_semi', default='dataset/preprocessed_data/training_semi_balance', help='Saving path for train_data_semi (used for pretraining)')
parser.add_argument('--train_data', default='dataset/preprocessed_data/training_balance', help='Saving path for train_data (used for fine-tuning)')
parser.add_argument('--valid_data', default='dataset/preprocessed_data/validation_balance', help='Saving path for valid_data')
parser.add_argument('--test_data', default='dataset/preprocessed_data/testing_balance', help='Saving path for test data (balanced)')
parser.add_argument('--test_data_imb', default='dataset/preprocessed_data/testing_imbalance', help='Saving path for test data (imbalanced)')

args = parser.parse_args()
print('Preprocessing started.....')
print()

# load lab and patient data, and merge these 2 dataframes
patient_data = pd.read_csv(args.patient_data).replace({'F': 0, 'M': 1})
lab_data = pd.read_csv(args.lab_data).drop(columns=['Bacteria', 'GLU', 'NIT', 'PRO']).rename(columns = {'CREA_x': 'CREA_blood','WBC_x': 'WBC_blood', 'CREA_y': 'CREA_urine', 'WBC_y': 'WBC_urine'})
data = lab_data.merge(patient_data[['病歷號', '透析日期', '換腎日期']], how='left', on='病歷號')

# Exclude kidney transplant and dialysis patient data
col_names = ['CREA_blood', 'BUN', 'Hb', 'NA', 'K', 'CA', 'P', 'ACR', 'PCR'] # 9 variable we choose

data[['檢查日', '透析日期', '換腎日期']] = data[['檢查日', '透析日期', '換腎日期']].apply(pd.to_datetime, errors='coerce')
data = data[(data['檢查日'] < data['換腎日期']) | (data['換腎日期'].isnull())]
data = data[(data['檢查日'] < data['透析日期']) | (data['透析日期'].isnull())]

data = data.replace({'SG': {'Negative(<2)': np.nan, 'S1': np.nan}, 'pH': {'Negative': np.nan}})
for col_name in data.select_dtypes(['object']).columns:
    data[col_name] = data[col_name].astype(str).str.extract('([-+]?\d*\.?\d+)').apply(pd.to_numeric, errors='coerce')

# Obtain the mean of each variable per month and pad the missing months
data['Y&M'] = data['檢查日'].dt.to_period('M').dt.to_timestamp()
data = data.drop(['檢查日', '透析日期', '換腎日期'], axis=1)

agg_dict = {}
agg_dict.update(dict.fromkeys(col_names, 'mean'))
data_monthly = data.groupby(['病歷號', 'Y&M'], as_index=False).agg(agg_dict)
data_monthly_padding = data_monthly.set_index('Y&M').groupby(['病歷號'])[data_monthly.columns[2:]].apply(lambda x: x.asfreq('MS', fill_value=np.nan)).reset_index()

# Use CKD-EPI formula to calculate eGFR
final_res = data_monthly_padding[['病歷號', 'Y&M', 'CREA_blood']]
final_res = final_res.merge(patient_data[['病歷號', 'Birthday', 'Sex']], how='left', on='病歷號')
final_res['Birthday'] = pd.to_datetime(final_res['Birthday']).dt.to_period('M').dt.to_timestamp()
final_res['age'] = (final_res['Y&M'] - final_res['Birthday'])/np.timedelta64(1, 'Y')
final_res['CKD-EPI-eGFR'] = final_res.apply(CKD_EPI, axis=1)
final_res = final_res.drop(['CREA_blood', 'Birthday', 'Sex'], axis=1)

data_monthly_padding = pd.concat([data_monthly_padding, final_res[['CKD-EPI-eGFR']]], axis=1)

# Take past and future time steps for each data entry 
final_res = getFutureList('CKD-EPI-eGFR', args.FutureTSs, data_monthly_padding, final_res)

col_names = list(col_names) + ['CKD-EPI-eGFR']
for col_name in col_names:
    print(col_name)
    final_res = getPastList(col_name, args.PastTSs, args.MPerTS, 'mean', data_monthly_padding, final_res)

final_res = final_res.groupby('病歷號', group_keys=False).apply(lambda group: group.iloc[args.PastTSs-1:, :])  # remove every patient first-year data to avoid too many missing value 

col_names.remove('CKD-EPI-eGFR')
final_res['combined'] = final_res.apply(lambda x: np.array(list([x[col_name] for col_name in col_names])).T.tolist(), axis=1)

# Exclude patient data with current eGFR < 60 and define GT
GT = f'if_drop{int(100 * args.drop_ratio)}%'
res_for_semi = final_res[(~final_res['future_eGFR'].notna()) & (final_res['current_eGFR'] <= 60)]
res_for_supv = final_res[(final_res['future_eGFR'].notna()) & (final_res['current_eGFR'] <= 60)]

res_for_semi[GT] = -1 # -1 means without label, 1 means drop rapidly, 0 means normal
res_for_supv[GT] = np.where((res_for_supv['current_eGFR'] - res_for_supv['future_eGFR'])/res_for_supv['current_eGFR'] > args.drop_ratio, 1, 0)

final_res = pd.concat([res_for_supv, res_for_semi]).sort_values(by=['病歷號', 'Y&M']).reset_index(drop=True)

print()
print('ratio of drop rapidly: ', round(len(final_res[final_res['if_drop30%'] == 1]) / len(res_for_supv), 3))
print('ratio of normal: ', round(len(final_res[final_res['if_drop30%'] == 0]) / len(res_for_supv), 3))

# Split preprocessed dataset into train, valid, test 
cut_1 = final_res[final_res['病歷號'] == args.cut1_PID].head(1).index.item() 
cut_2 = final_res[final_res['病歷號'] == args.cut2_PID].head(1).index.item()
train_data, test_data, valid_data = final_res[:cut_1], final_res[cut_1:cut_2], final_res[cut_2:]

# Under sampling
train_pos = train_data[train_data[GT] == 1]
train_neg = train_data[train_data[GT] == 0].sample(n=len(train_pos), random_state=100)
train_semi = train_data[train_data[GT] == -1]
train_data_balance = pd.concat([train_pos, train_neg]).sort_values(by=['病歷號', 'Y&M']).reset_index(drop=True)
train_data_semi_balance = pd.concat([train_pos, train_neg, train_semi]).sort_values(by=['病歷號', 'Y&M']).reset_index(drop=True)

valid_pos = valid_data[valid_data[GT] == 1]
valid_neg = valid_data[valid_data[GT] == 0].sample(n=len(valid_pos), random_state=100)
valid_semi = valid_data[valid_data[GT] == -1]
valid_data_balance = pd.concat([valid_pos, valid_neg]).sort_values(by=['病歷號', 'Y&M']).reset_index(drop=True)

test_pos = test_data[test_data[GT] == 1]
test_neg = test_data[test_data[GT] == 0].sample(n=len(test_pos), random_state=100)
test_semi = test_data[test_data[GT] == -1]
test_data_balance = pd.concat([test_pos, test_neg]).sort_values(by=['病歷號', 'Y&M']).reset_index(drop=True)
test_data_imbalance = test_data[test_data[GT] != -1]

train_data_semi_mergePT = train_data_semi_balance.merge(patient_data, how='left', on='病歷號') # training set (balanced pos and neg + no label) => for pretraining
train_data_mergePT = train_data_balance.merge(patient_data, how='left', on='病歷號')           # training set (balanced pos and neg) => for fine tuning
test_data_mergePT = test_data_balance.merge(patient_data, how='left', on='病歷號')             # testing  set (balanced pos and neg)
test_data_imb_mergePT = test_data_imbalance.merge(patient_data, how='left', on='病歷號')       # testing  set (imbalanced pos and neg)
valid_data_mergePT = valid_data_balance.merge(patient_data, how='left', on='病歷號')           # valid    set (balanced pos and neg)

# Save dataset into pickle file
import pickle
GT_description = f'{int(args.FutureTSs/12)}yearDrop{int(100 * args.drop_ratio)}%'
if(args.impute_strategy != ''):
    GT_description = f'{GT_description}_{args.impute_strategy}'
    
with open(f'{args.train_data_semi}_{GT_description}.pickle', 'wb') as f:
    pickle.dump((train_data_semi_mergePT['combined'].tolist(), train_data_semi_mergePT[['Sex', 'age']].values.tolist(), 
                 train_data_semi_mergePT[GT].tolist(), train_data_semi_mergePT['future_eGFR'].tolist(), train_data_semi_mergePT['current_eGFR'].tolist()), f)

with open(f'{args.train_data}_{GT_description}.pickle', 'wb') as f:
    pickle.dump((train_data_mergePT['combined'].tolist(), train_data_mergePT[['Sex', 'age']].values.tolist(), 
                 train_data_mergePT[GT].tolist(), train_data_mergePT['future_eGFR'].tolist(), train_data_mergePT['current_eGFR'].tolist()), f)

with open(f'{args.valid_data}_{GT_description}.pickle', 'wb') as f:
    pickle.dump((valid_data_mergePT['combined'].tolist(), valid_data_mergePT[['Sex', 'age']].values.tolist(),
                 valid_data_mergePT[GT].tolist(), valid_data_mergePT['future_eGFR'].tolist(), valid_data_mergePT['current_eGFR'].tolist()), f)
    
with open(f'{args.test_data}_{GT_description}.pickle', 'wb') as f:
    pickle.dump((test_data_mergePT['combined'].tolist(), test_data_mergePT[['Sex', 'age']].values.tolist(), 
                 test_data_mergePT[GT].tolist(), test_data_mergePT['future_eGFR'].tolist(), test_data_mergePT['current_eGFR'].tolist()), f)

with open(f'{args.test_data_imb}_{GT_description}.pickle', 'wb') as f:
    pickle.dump((test_data_imb_mergePT['combined'].tolist(), test_data_imb_mergePT[['Sex', 'age']].values.tolist(), 
                 test_data_imb_mergePT[GT].tolist(), test_data_imb_mergePT['future_eGFR'].tolist(), test_data_imb_mergePT['current_eGFR'].tolist()), f)

print()
print('Preprocessing completed.....')
