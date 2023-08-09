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
    def collate_fn(batch, input_format):
        
        x_result = []
        y_result = []
        f_result = []
        PT_basic_result = []
        
        for b, l in batch:
            x_result.append(b[0])
            PT_basic_result.append(b[1])
            f_result.append((~np.isnan(np.array(b[0]))).astype(dtype=float).tolist()) # get only missing/exist matrix 
            y_result.append(l[2])
        
        x_result = torch.tensor(x_result)
        if(input_format == 'Last3TS'):
            x_result = x_result[:,-3:,:].nanmean(dim=1)
        x_result[torch.isnan(x_result)] = 0
        PT_basic_result = torch.tensor(PT_basic_result).type(torch.float32)
        f_result = torch.tensor(f_result)
        y_result = torch.tensor(y_result)

        return (x_result, PT_basic_result, f_result, y_result)