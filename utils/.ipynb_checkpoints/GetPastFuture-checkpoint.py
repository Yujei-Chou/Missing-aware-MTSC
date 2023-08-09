import pandas as pd
import numpy as np

def getPastList(col_name, m_cnt, m_win, agg_type, data_monthly_padding, final_res):
    sub_col_names = []
    sub_col_df = pd.DataFrame()
    
    for i in range(m_cnt):
        sub_col_names.append(col_name + str(m_cnt - i - 1))
        sub_col_df[col_name + str(i)] = data_monthly_padding.groupby(['病歷號'], as_index=False)[col_name].shift(i)
    
    if(col_name == 'CKD-EPI-eGFR'):
        final_res['current_eGFR'] = sub_col_df[sub_col_names[-3:]].mean(axis=1)
        final_res['eGFR'] = sub_col_df[sub_col_names].groupby(np.arange(len(sub_col_names))//m_win, axis=1).mean().values.tolist()      
    else:
        final_res[col_name] = sub_col_df[sub_col_names].groupby(np.arange(len(sub_col_names))//m_win, axis=1).mean().values.tolist()

    return final_res

def getFutureList(col_name, m_cnt, data_monthly_padding, final_res):
    sub_col_names = []
    sub_col_df = pd.DataFrame()
    
    for i in range(m_cnt):
        sub_col_names.append(col_name + str(i+1))
        sub_col_df[col_name + str(i+1)] = data_monthly_padding.groupby(['病歷號'], as_index=False)[col_name].shift(-(i+1))

    final_res['future_eGFR'] = sub_col_df[sub_col_names[-3:]].mean(axis=1)
    # final_res['future_eGFR_List'] = sub_col_df[sub_col_names].values.tolist()
    
    return final_res

        