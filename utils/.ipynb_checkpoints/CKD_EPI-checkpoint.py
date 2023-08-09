import pandas as pd

def CKD_EPI(df):
    if(df['Sex'] == 1 and df['CREA_blood'] <= 0.9):
        return 141 * ((df['CREA_blood'] / 0.9)**(-0.411)) * (0.993 ** df['age'])
    elif(df['Sex'] == 1 and df['CREA_blood'] > 0.9):
        return 141 * ((df['CREA_blood'] / 0.9)**(-1.209)) * (0.993 ** df['age'])
    elif(df['Sex'] == 0 and df['CREA_blood'] <= 0.7):
        return 144 * ((df['CREA_blood'] / 0.7)**(-0.329)) * (0.993 ** df['age'])
    else:
        return 144 * ((df['CREA_blood'] / 0.7)**(-1.209)) * (0.993 ** df['age'])