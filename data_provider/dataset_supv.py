import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, visits,  labels):
        self.visits = visits
        self.labels = labels

    def __len__(self):
        return len(self.visits)

    def __getitem__(self, index):
        visit = self.visits[index]
        label = self.labels[index]
        
        return visit, label

    @staticmethod
    def collate_fn(batch, norm_type):
        
        x_result = []
        y_result = []
        f_result = []
        PT_basic_result = []
        if(norm_type == 'Standarlize'): missing_impute = 0
        else: missing_impute = -1        
        
        for b, l in batch:
            x_result.append(b[0])         # lab data (9 variables) 
            PT_basic_result.append(b[1])  # patient data (age, gender)
            f_result.append((~np.isnan(np.array(b[0]))).astype(dtype=float).tolist()) # get missing mask 
            y_result.append(l[2])         # Ground Truth
        
        # turn to tensor type
        x_result = torch.tensor(x_result)
        x_result[torch.isnan(x_result)] = missing_impute
        PT_basic_result = torch.tensor(PT_basic_result).type(torch.float32)
        f_result = torch.tensor(f_result)
        y_result = torch.tensor(y_result)


        return (x_result, PT_basic_result, f_result, y_result)
        