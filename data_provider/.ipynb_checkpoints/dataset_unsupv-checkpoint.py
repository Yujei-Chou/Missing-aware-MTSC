import torch
from torch.utils.data import Dataset
import numpy as np
import random

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
        f_result = []
        PT_basic_result = []
        target_result = []
        mask_result = []
        if(norm_type == 'Standarlize'): missing_impute = 0
        else: missing_impute = -1
        mask_ratio = 0.15
        
        for b, l in batch:
            target_result.append(b[0])
            PT_basic_result.append(b[1])
            
            notNan_idxs = np.argwhere(~np.isnan(np.array(b[0])))           
            random_mask_idxs = random.sample([i for i in range (len(notNan_idxs))], int(len(notNan_idxs)*mask_ratio)) # artificially masking 15%
            mask = np.zeros((12, 9))
            x = np.array(b[0])
            missing_mask = (~np.isnan(np.array(b[0]))).astype(dtype=float)
            for idx in notNan_idxs[random_mask_idxs]:
                mask[idx[0], idx[1]] = 1
                x[idx[0], idx[1]] = missing_impute
                missing_mask[idx[0], idx[1]] = 0
            
            
            mask_result.append(mask)
            x_result.append(x)
            f_result.append(missing_mask)

            
        # turn to tensor type
        x_result = torch.tensor(x_result).type(torch.float32)
        x_result[torch.isnan(x_result)] = missing_impute
        PT_basic_result = torch.tensor(PT_basic_result).type(torch.float32)
        f_result = torch.tensor(f_result).type(torch.float32)
        mask_result = torch.tensor(mask_result).type(torch.ByteTensor)
        target_result = torch.tensor(target_result)
        target_result[torch.isnan(target_result)] = missing_impute

        
        # f_result: missing_mask, mask_result: masking_mask, target_result: observing_mask
        return (x_result, PT_basic_result, f_result, mask_result, target_result)
        

