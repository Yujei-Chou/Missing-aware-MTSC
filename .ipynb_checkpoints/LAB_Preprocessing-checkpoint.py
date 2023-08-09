import os
import pyodbc
import pandas as pd
from functools import reduce
import warnings

warnings.filterwarnings('ignore')


data_arr = []

for filename in os.listdir('dataset/IRB_data/Labdata'):
    if(filename.endswith('.mdb')):
        print(filename)
        conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=dataset/IRB_data/Labdata/' + f'{filename};')
        data_arr.append(pd.read_sql('select * from  Labdata', conn))
    

new_data = pd.concat(data_arr)
cols = ['OrderCode', 'OrderName', 'ItemCode', 'ItemName', 'TestResult', 'Unit']
new_data[cols] = new_data[cols].apply(lambda x: x.str.strip())
new_data['檢查日'] = pd.to_datetime(new_data['SampleTime']).dt.date
new_data = new_data.drop(columns = ['SUID', 'RequestNo', 'OrderCode', 'SampleTime'])
new_data['ChartNo'] = new_data['ChartNo'].apply(pd.to_numeric)
print(new_data)


code_book = pd.read_csv('dataset/IRB_data/檢驗項目對照表(精簡版).csv', encoding='big5')
code_book[['項目名稱', '項目代碼']] = code_book[['項目名稱', '項目代碼']].apply(lambda x: x.str.strip())
code_book.at[3, '項目名稱'] = 'NA'

# Get urine coding book
urine = code_book[code_book['血液/尿液'] == 1].groupby('項目名稱')['項目代碼'].apply(list).reset_index()
urine_df_list = []

for index, row in urine.iterrows():
    urine_df = new_data[new_data['ItemCode'].isin(row['項目代碼'])][['ChartNo', '檢查日', 'TestResult']].rename({'ChartNo': '病歷號', 'TestResult': row['項目名稱']}, axis=1).drop_duplicates()
    urine_df = urine_df.groupby(['病歷號', '檢查日']).last().reset_index()
    urine_df_list.append(urine_df)
    
urine_res = reduce(lambda  left,right: pd.merge(left,right,on=['病歷號','檢查日'], how='outer'), urine_df_list).sort_values(by=['病歷號', '檢查日']).reset_index(drop=True) 

# Get blood coding book
blood = code_book[code_book['血液/尿液'] == 0].groupby('項目名稱')['項目代碼'].apply(list).reset_index()
blood_df_list = []

for index, row in blood.iterrows():
    blood_df = new_data[new_data['ItemCode'].isin(row['項目代碼'])][['ChartNo', '檢查日', 'TestResult']].rename({'ChartNo': '病歷號', 'TestResult': row['項目名稱']}, axis=1).drop_duplicates()
    blood_df = blood_df.groupby(['病歷號', '檢查日']).last().reset_index()
    blood_df_list.append(blood_df)
    
blood_res = reduce(lambda  left,right: pd.merge(left,right,on=['病歷號','檢查日'], how='outer'), blood_df_list).sort_values(by=['病歷號', '檢查日']).reset_index(drop=True)

# Merge both urine and blood data
all_res = pd.merge(blood_res, urine_res, on=['病歷號', '檢查日'], how='outer').sort_values(by=['病歷號', '檢查日'])
print(all_res)
all_res.to_csv('dataset/lab_data.csv', index=False, encoding='utf-8-sig')