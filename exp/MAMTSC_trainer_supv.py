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
        self.optim = Adam(self.get_pretrain_params(self.MAMTSC), lr=lr)
        self.criterion = FocalLoss(gamma=0)
                
        self.train_loss_arr = []
        self.valid_loss_arr = []
        self.train_acc_arr = []
        self.valid_acc_arr = []
        
        self.best_auroc = 0.0
        self.best_auprc = 0.0
        self.best_f1 = 0.0
        self.best_acc = 0.0


    def get_pretrain_params(self, model):
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        
        return params_to_update
        
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
        correct_train, total_train = 0, 0
        self.MAMTSC.train()
        
        with tqdm(self.train_dataloader, desc = 'Train', file = sys.stdout) as iterator:
            step = 0

            for padding_input, PT_input, freq_input, input_labels in iterator:
                padding_input, PT_input = Variable(padding_input).to(self.device), Variable(PT_input).to(self.device)
                freq_input = Variable(freq_input).to(self.device)
                input_labels = Variable(input_labels).to(self.device)
                
                predict_output = self.MAMTSC(padding_input, PT_input, freq_input)
                CF_output = predict_output.squeeze(1)
                
                threshold = Variable(torch.Tensor([0.5])).to(self.device)
                binary_labels = (CF_output > threshold).float().to(self.device)
                correct_train += (binary_labels == input_labels).sum()
                total_train += input_labels.size(0)
                
                loss = self.criterion(CF_output, input_labels)
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                train_loss += loss.item()
                step += 1
            
            
            acc = correct_train/total_train
            print('train loss: %.3f | train acc: %.3f' % \
                      (train_loss/step, acc))
                
        
        self.train_loss_arr.append(train_loss/step)
        self.train_acc_arr.append(100*acc.item())
                
        


    def validate(self, epoch):
        
        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
        valid_loss = 0.0
        total_valid = 0
        total_labels = np.array([])
        total_preds = np.array([])        
        self.MAMTSC.eval()
        
        with tqdm(self.validate_dataloader, desc = 'Valid', file = sys.stdout) as iterator:
            step = 0
            for padding_input, PT_input, freq_input, input_labels in iterator:
                padding_input, PT_input = Variable(padding_input).to(self.device), Variable(PT_input).to(self.device)
                freq_input = Variable(freq_input).to(self.device)
                input_labels = Variable(input_labels).to(self.device)
                
                predict_output = self.MAMTSC(padding_input, PT_input, freq_input)
                CF_output = predict_output.squeeze(1)
                
                threshold = Variable(torch.Tensor([0.5])).to(self.device)
                binary_labels = (CF_output > threshold).float().to(self.device)
                
                
                ############################## code for auc #################################
                labels_temp = input_labels.cpu().numpy()
                preds_temp  = CF_output.detach().cpu().numpy()                
                total_labels = np.concatenate((total_labels, labels_temp))
                total_preds = np.concatenate((total_preds, preds_temp))                
                
                total_valid += input_labels.size(0)
                
                temp_TP, temp_FP, temp_TN, temp_FN = self.evaluate(binary_labels, input_labels)
                total_TP += temp_TP
                total_FP += temp_FP
                total_TN += temp_TN
                total_FN += temp_FN
                ##############################################################################
                

                loss = self.criterion(CF_output, input_labels)
                
                valid_loss += loss.item()
                step += 1
        

            fpr, tpr, thresholds = metrics.roc_curve(total_labels, total_preds, pos_label=1)
            precision, recall, thresholds = metrics.precision_recall_curve(total_labels, total_preds)
            
            auroc = metrics.auc(fpr, tpr)
            auprc = metrics.auc(recall, precision)
            acc = (total_TP + total_TN) / total_valid
            p, r, f1 = self.precision_recall_f1(total_TP, total_FP, total_FN)

            print('valid loss: %.3f | valid acc: %.3f | auroc: %.3f | auprc: %.3f | precision: %.3f | recall: %.3f | f1: %.3f | TP: %d | FP: %d | TN: %d | FN: %d ' % \
                      (valid_loss/step, acc, auroc, auprc, p, r, f1, total_TP, total_FP, total_TN, total_FN))
            

            if(acc >= self.best_acc): # save model with best accuracy
                self.best_acc = acc
                torch.save(self.MAMTSC.state_dict(), f'{self.model_name}.pth')
                print('Best!!')
                
        self.valid_loss_arr.append(valid_loss/step)
        self.valid_acc_arr.append(100*acc.item())
    


    def test(self, epoch):
        total_TP, total_FP, total_TN, total_FN = 0, 0, 0, 0
        test_loss = 0.0
        total_test = 0
        total_labels = np.array([])
        total_preds = np.array([])          
        self.MAMTSC.eval()
        res = []
        
        
        with tqdm(self.test_dataloader, desc = 'test', file = sys.stdout) as iterator:
            step = 0
            for padding_input, PT_input, freq_input, input_labels in iterator:
                padding_input, PT_input = Variable(padding_input).to(self.device), Variable(PT_input).to(self.device)
                freq_input = Variable(freq_input).to(self.device)
                input_labels = Variable(input_labels).to(self.device)
                
                predict_output = self.MAMTSC(padding_input, PT_input, freq_input)
                CF_output = predict_output.squeeze(1)

                threshold = Variable(torch.Tensor([0.5])).to(self.device)
                binary_labels = (CF_output > threshold).float().to(self.device)

                ############################## code for auc #################################
                labels_temp = input_labels.cpu().numpy()
                preds_temp  = CF_output.detach().cpu().numpy()                
                total_labels = np.concatenate((total_labels, labels_temp))
                total_preds = np.concatenate((total_preds, preds_temp))                
                
                total_test += input_labels.size(0)
                
                temp_TP, temp_FP, temp_TN, temp_FN = self.evaluate(binary_labels, input_labels)
                total_TP += temp_TP
                total_FP += temp_FP
                total_TN += temp_TN
                total_FN += temp_FN
                ##############################################################################                
                

                
                res = res + list(zip(input_labels.cpu().numpy(), binary_labels.detach().cpu().numpy(), np.round(CF_output.detach().cpu().numpy(), 3)))
                # step += 1

            fpr, tpr, thresholds = metrics.roc_curve(total_labels, total_preds, pos_label=1)
            precision, recall, thresholds = metrics.precision_recall_curve(total_labels, total_preds)
            
            auroc = metrics.auc(fpr, tpr)
            auprc = metrics.auc(recall, precision)
            acc = (total_TP + total_TN) / total_test
            p, r, f1 = self.precision_recall_f1(total_TP, total_FP, total_FN)
            specificity = total_TN/(total_TN+total_FP)

            print('test acc: %.3f | auroc: %.3f | auprc: %.3f | precision: %.3f | sensitivity: %.3f | specificity: %.3f | f1: %.3f | TP: %d | FP: %d | TN: %d | FN: %d ' % \
                      (acc, auroc, auprc, p, r, specificity, f1, total_TP, total_FP, total_TN, total_FN))
            

        
        return res



    def plot_figure(self):
        plt.figure()
        
        plt.plot(self.train_loss_arr) # plot your loss
        plt.plot(self.valid_loss_arr)
        
        plt.title('Loss')
        plt.ylabel('loss'), plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc = 'upper left')
        plt.show()


        plt.figure()
        
        plt.plot(self.train_acc_arr) # plot your training accuracy
        plt.plot(self.valid_acc_arr) # plot your testing accuracy
        
        plt.title('Accuracy')
        plt.ylabel('acc (%)'), plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc = 'upper left')
        plt.show()        