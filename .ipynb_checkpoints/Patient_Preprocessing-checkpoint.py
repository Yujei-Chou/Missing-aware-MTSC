import pyodbc
import pandas as pd
from functools import reduce
import warnings

warnings.filterwarnings('ignore')

# Get ICD10 code (data inside and outside the hospital)
conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=dataset/IRB_data/ICDcode/ICDCode10.mdb;')
INP_ICD10 = pd.read_sql('select * from  INP', conn)
INP_ICD10['InHospital'] = 1
INP_ICD10 = INP_ICD10[['ChartNo', 'InDate', 'ICDCode', 'InHospital', 'Sex', 'Birthday']].rename({'ChartNo': '病歷號', 'InDate': '檢查日'}, axis=1)

OPD_ICD10 = pd.read_sql('select * from  OPD', conn)
OPD_ICD10['InHospital'] = 0
OPD_ICD10 = OPD_ICD10[['ChartNo', 'VisitDate', 'ICDCode', 'InHospital', 'Sex', 'Birthday']].rename({'ChartNo': '病歷號', 'VisitDate': '檢查日'}, axis=1)

all_ICD10 = pd.concat([INP_ICD10, OPD_ICD10])
all_ICD10['病歷號'] = all_ICD10['病歷號'].apply(pd.to_numeric)
all_ICD10 = all_ICD10.sort_values(by=['病歷號', '檢查日']).drop_duplicates()


# Get ICD9 code (data inside and outside the hospital)
conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=dataset/IRB_data/ICDcode/ICDCode9.mdb;')
INP_ICD9 = pd.read_sql('select * from  INP', conn)
INP_ICD9['InHospital'] = 1
INP_ICD9 = INP_ICD9[['ChartNo', 'InDate', 'ICDCode', 'InHospital', 'Sex', 'Birthday']].rename({'ChartNo': '病歷號', 'InDate': '檢查日'}, axis=1)

OPD_ICD9 = pd.read_sql('select * from  OPD', conn)
OPD_ICD9['InHospital'] = 0
OPD_ICD9 = OPD_ICD9[['ChartNo', 'VisitDate', 'ICDCode', 'InHospital', 'Sex', 'Birthday']].rename({'ChartNo': '病歷號', 'VisitDate': '檢查日'}, axis=1)

all_ICD9 = pd.concat([INP_ICD9, OPD_ICD9])
all_ICD9['病歷號'] = all_ICD9['病歷號'].apply(pd.to_numeric)
all_ICD9 = all_ICD9.sort_values(by=['病歷號', '檢查日']).drop_duplicates()


# Get kidney transplant dates (ICD9: V420, ICD10: Z940)
kidney_trans = pd.concat([all_ICD10[all_ICD10['ICDCode'] == 'Z940'], all_ICD9[all_ICD9['ICDCode'] == 'V420']]).sort_values(by=['病歷號', '檢查日'])
kidney_trans = kidney_trans.groupby(['病歷號']).first().reset_index().rename(columns={'檢查日': '換腎日期'})[['病歷號', 'Sex', 'Birthday', '換腎日期']]

# Get dialysis dates (ICD9: 586,  ICD10: N186)
dialysis = pd.concat([all_ICD10[all_ICD10['ICDCode'] == 'N186'], all_ICD9[all_ICD9['ICDCode'] == '586']]).sort_values(by=['病歷號', '檢查日'])
dialysis = dialysis.groupby(['病歷號']).first().reset_index().rename(columns={'檢查日': '透析日期'})[['病歷號', 'Sex', 'Birthday', '透析日期']]

# Merge all important patient basic data: Sex, Birthday, Dialysis date, Kidney transplant date 
patient_basic_special = pd.merge(dialysis, kidney_trans, how='outer', on=['病歷號', 'Sex', 'Birthday'])

all_ICD = pd.concat([all_ICD9, all_ICD10])
patient_basic_all = all_ICD.groupby('病歷號').first().reset_index()[['病歷號', 'Sex', 'Birthday']]
patient_basic_all = pd.merge(patient_basic_special, patient_basic_all, how='outer', on=['病歷號', 'Sex', 'Birthday']).sort_values('病歷號')
print(patient_basic_all)
patient_basic_all.to_csv('dataset/patient_data.csv', index=False, encoding='utf-8-sig')