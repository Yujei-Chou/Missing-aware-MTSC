import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn import metrics
from focal_loss.focal_loss import FocalLoss


class MAMTSC_trainer:
    def __init__(self, MAMTSC_model, model_name, train_dataloader, validate_dataloader, test_dataloader, with_cuda=0,lr=0.001):
        super().__init__()
        self.device = torch.device("cuda:0" if with_cuda==1 else "cpu")
        self.MAMTSC = MAMTSC_model.to(self.device)
        self.model_name = model_name
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader
        self.optim = Adam(self.MAMTSC.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.train_loss_arr = []
        self.valid_loss_arr = []
    
        self.best_loss = 1000.0


        
    def evaluate(self, pred, label):
        
        TP = (label == 1) & (pred == 1)
        FP = (label == 0) & (pred == 1)
        TN = (label == 0) & (pred == 0)
        FN = (label == 1) & (pred == 0)
        
        TP, FP, TN, FN = torch.count_nonzero(TP), torch.count_nonzero(FP), torch.count_nonzero(TN), torch.count_nonzero(FN)
        return TP, FP, TN, FN
        
    
    def precision_recall_f1(self, TP, FP, FN):

        if((TP == 0) & (FP == 0) & (FN == 0)):
            precision, recall = 1, 1
        elif(((TP + FP)==0) | ((TP + FN) == 0)):
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
        
        if((precision + recall) == 0):
            f1_score = 0
        else:
            f1_score = (2*precision*recall) / (precision + recall)
        
        return precision, recall, f1_score


    def train(self, epoch):
        
        train_loss = 0.0
        train_MPT_loss, train_OPT_loss = 0.0, 0.0
        self.MAMTSC.train()
        
        with tqdm(self.train_dataloader, desc = 'Train', file = sys.stdout) as iterator:
            step = 0

            for lab_input, PT_input, freq_input, mask_input, target_input in iterator:
                lab_input, PT_input = Variable(lab_input).to(self.device), Variable(PT_input).to(self.device)
                freq_input, mask_input = Variable(freq_input).to(self.device), Variable(mask_input).to(self.device)
                input_labels = Variable(target_input).to(self.device)
                observe_input = Variable(freq_input).type(torch.ByteTensor).to(self.device)


                predict_output = self.MAMTSC(lab_input, PT_input, freq_input)
                masked_impute_true = torch.masked_select(input_labels, mask_input)
                masked_impute_pred = torch.masked_select(predict_output, mask_input)
                
                observed_reconstruct_true = torch.masked_select(input_labels, observe_input)
                observed_reconstruct_pred = torch.masked_select(predict_output, observe_input)
                
                MPT_loss = self.criterion(masked_impute_pred, masked_impute_true)
                OPT_loss = self.criterion(observed_reconstruct_pred, observed_reconstruct_true)
                loss = MPT_loss + OPT_loss
                                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                train_loss += loss.item()
                train_MPT_loss += MPT_loss.item()
                train_OPT_loss += OPT_loss.item()
                step += 1
            
            
            print('train loss: %.5f | MPT loss: %.5f | OPT loss: %.5f' % (train_loss/step, train_MPT_loss/step, train_OPT_loss/step))
            
        
        self.train_loss_arr.append(train_loss/step)
                
        


    def validate(self, epoch):
        
        valid_loss = 0.0
        valid_MPT_loss, valid_OPT_loss = 0.0, 0.0
        self.MAMTSC.eval()
        
        with tqdm(self.validate_dataloader, desc = 'Valid', file = sys.stdout) as iterator:
            step = 0
            for lab_input, PT_input, freq_input, mask_input, target_input in iterator:
                lab_input, PT_input = Variable(lab_input).to(self.device), Variable(PT_input).to(self.device)
                freq_input, mask_input = Variable(freq_input).to(self.device), Variable(mask_input).to(self.device)
                input_labels = Variable(target_input).to(self.device)
                observe_input = Variable(freq_input).type(torch.ByteTensor).to(self.device)
                
                predict_output = self.MAMTSC(lab_input, PT_input, freq_input)
                masked_impute_true = torch.masked_select(input_labels, mask_input)
                masked_impute_pred = torch.masked_select(predict_output, mask_input)
                
                observed_reconstruct_true = torch.masked_select(input_labels, observe_input)
                observed_reconstruct_pred = torch.masked_select(predict_output, observe_input)
                
                MPT_loss = self.criterion(masked_impute_pred, masked_impute_true)
                OPT_loss = self.criterion(observed_reconstruct_pred, observed_reconstruct_true)
                loss = MPT_loss + OPT_loss
                
                valid_loss += loss.item()
                valid_MPT_loss += MPT_loss.item()
                valid_OPT_loss += OPT_loss.item()
                step += 1
        


            print('valid loss: %.5f | MPT loss: %.5f | OPT loss: %.5f' % (valid_loss/step, valid_MPT_loss/step, valid_OPT_loss/step))
            
            
            if((valid_loss/step) < self.best_loss):
                self.best_loss = (valid_loss/step)
                torch.save(self.MAMTSC.state_dict(), f'{self.model_name}.pth')
                print('Best!!')
                
        self.valid_loss_arr.append(valid_loss/step)
    


    def test(self, epoch):
        test_loss = 0.0
        test_MPT_loss, test_OPT_loss = 0.0, 0.0
        self.MAMTSC.eval()
        res = []
        
        with tqdm(self.test_dataloader, desc = 'test', file = sys.stdout) as iterator:
            step = 0
            for lab_input, PT_input, freq_input, mask_input, target_input in iterator:
                lab_input, PT_input = Variable(lab_input).to(self.device), Variable(PT_input).to(self.device)
                freq_input, mask_input = Variable(freq_input).to(self.device), Variable(mask_input).to(self.device)
                input_labels = Variable(target_input).to(self.device)
                observe_input = Variable(freq_input).type(torch.ByteTensor).to(self.device)
                
                predict_output = self.MAMTSC(lab_input, PT_input, freq_input)
                masked_impute_true = torch.masked_select(input_labels, mask_input)
                masked_impute_pred = torch.masked_select(predict_output, mask_input)
                
                observed_reconstruct_true = torch.masked_select(input_labels, observe_input)
                observed_reconstruct_pred = torch.masked_select(predict_output, observe_input)
                
                MPT_loss = self.criterion(masked_impute_pred, masked_impute_true)
                OPT_loss = self.criterion(observed_reconstruct_pred, observed_reconstruct_true)
                loss = MPT_loss + OPT_loss
                

                test_loss += loss.item()
                test_MPT_loss += MPT_loss.item()
                test_OPT_loss += OPT_loss.item()
                step += 1


            print('test loss: %.5f | MPT loss: %.5f | OPT loss: %.5f' % (test_loss/step, test_MPT_loss/step, test_OPT_loss/step))
            
        
        
        return res



    def plot_figure(self):
        plt.figure()
        
        plt.plot(self.train_loss_arr) # plot your loss
        plt.plot(self.valid_loss_arr)
        
        plt.title('Loss')
        plt.ylabel('loss'), plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc = 'upper left')
        plt.show()
        