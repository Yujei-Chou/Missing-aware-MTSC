import pandas as pd
import numpy as np


# Normalize the laboratory data
def lab_Norm(norm_type, seq_len, base_combine, train_combine, valid_combine, test_combine):
    feats = ['CREA_blood', 'BUN', 'Hb', 'NA', 'K', 'CA', 'P', 'ACR', 'PCR']
    
    base_flatten = pd.DataFrame(np.array(base_combine).reshape(len(base_combine)*seq_len, len(feats)).tolist(), columns=feats)
    train_flatten = pd.DataFrame(np.array(train_combine).reshape(len(train_combine)*seq_len, len(feats)).tolist(), columns=feats)
    valid_flatten = pd.DataFrame(np.array(valid_combine).reshape(len(valid_combine)*seq_len, len(feats)).tolist(), columns=feats)
    test_flatten = pd.DataFrame(np.array(test_combine).reshape(len(test_combine)*seq_len, len(feats)).tolist(), columns=feats)
    

    if(norm_type == 'Standarlize'):
        train_Norm = (train_flatten - base_flatten.mean())/base_flatten.std()
        valid_Norm = (valid_flatten - base_flatten.mean())/base_flatten.std()
        test_Norm = (test_flatten - base_flatten.mean())/base_flatten.std()
    else:
        train_Norm = (train_flatten - base_flatten.min())/(base_flatten.max() - base_flatten.min())
        valid_Norm = (valid_flatten - base_flatten.min())/(base_flatten.max() - base_flatten.min())
        test_Norm = (test_flatten - base_flatten.min())/(base_flatten.max() - base_flatten.min())
        
    # Dataframe to 2D list
    train_combine_Norm = train_Norm.to_numpy().reshape((np.array(train_combine).shape[0], seq_len, len(feats))).tolist()
    valid_combine_Norm = valid_Norm.to_numpy().reshape((np.array(valid_combine).shape[0], seq_len, len(feats))).tolist()
    test_combine_Norm = test_Norm.to_numpy().reshape((np.array(test_combine).shape[0], seq_len, len(feats))).tolist()
    
    return (train_combine_Norm, valid_combine_Norm, test_combine_Norm)


# Normalize the patient data
def patient_Norm(norm_type, train_basic, valid_basic, test_basic):
    train_basic = pd.DataFrame(train_basic, columns=['Sex', 'age'])
    valid_basic = pd.DataFrame(valid_basic, columns=['Sex', 'age'])
    test_basic = pd.DataFrame(test_basic, columns=['Sex', 'age'])
    
    if(norm_type == 'Standarlize'):
        mean_val, std_val = train_basic['age'].mean(), train_basic['age'].std()
        train_basic['age'] = (train_basic['age'] - mean_val) / std_val
        valid_basic['age'] = (valid_basic['age'] - mean_val) / std_val
        test_basic['age'] = (test_basic['age'] - mean_val) / std_val
    else:
        min_val, max_val = train_basic['age'].min(), train_basic['age'].max()
        train_basic['age'] = (train_basic['age'] - min_val) / (max_val - min_val)
        valid_basic['age'] = (valid_basic['age'] - min_val) / (max_val - min_val)
        test_basic['age'] = (test_basic['age'] - min_val) / (max_val - min_val)
    
    
    return (train_basic.values.tolist(), valid_basic.values.tolist(), test_basic.values.tolist())