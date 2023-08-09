import torch
print('GPU info:')
print(f'if cuda is available: {torch.cuda.is_available()}')		# 查看GPU是否可用
print(f'the number of GPUs: {torch.cuda.device_count()}') 		# 查看GPU数量
print(f'the name of GPU: {torch.cuda.get_device_name()}')   	# 查看當前GPU設備名稱
print(f'the ID of GPU: {torch.cuda.current_device()}')		    # 查看當前GPU設備ID
print()
torch.cuda.set_device(0)

from utils.Normalize import lab_Norm, patient_Norm 
from copy import deepcopy
from torch.utils.data import DataLoader
import argparse
import os
import random
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# model define
parser = argparse.ArgumentParser()

parser.add_argument('--seq_len', type=int, default=12, help='sequence length')
parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')

# Transformer setting
parser.add_argument('--d_feedforawd', type=int, default=256, help='dimension of feedfoward')
parser.add_argument('--n_heads', type=int, default=16, help='num of heads') 
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--drop_p', type=float, default=0.1, help='trans drop probability')

# LSTM setting
parser.add_argument('--lstm_layers', type=int, default=4, help='number of lstm layer')
parser.add_argument('--lstm_drop_p', type=float, default=0.5, help='lstm drop probability')

# TCN setting
parser.add_argument('--tcn_layers', type=int, default=4, help='number of tcn layer')
parser.add_argument('--tcn_ksize', type=int, default=3, help='kernel size')
parser.add_argument('--tcn_drop_p', type=float, default=0.1, help='tcn drop probability')

# other setting
parser.add_argument('--horizon', type=int, default=1, help='forcast horizon')
parser.add_argument('--drop_ratio', type=float, default=0.3, help='drop ratio')
parser.add_argument('--if_FM', type=int, default=1, help='if use Fusion Module')
parser.add_argument('--model', type=str, default='MAMTSC', help='MAMTSC / LSTM / TCN')
parser.add_argument('--norm', type=str, default='Normalize', help='Standarlize / Normalize')
parser.add_argument('--finetune', action='store_true', default=False, help='if finetune')
parser.add_argument('--pretrain', action='store_true', default=False, help='if pretrain')
parser.add_argument('--linearprob', action='store_true', default=False, help='if linearprob')
parser.add_argument('--training_type', type=str, default='balance', help='balance / semi_balance')
parser.add_argument('--testing_type', type=str, default='balance', help='balance / imbalance / custom')
parser.add_argument('--testing_path', type=str, default='dataset/preprocessed_data/Experts_Questions/Sample100_form1', help='path for the pretrained weights')



# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

# GPU
parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--pretrain_path', type=str, default='weight/pretrain/Ours_1yearDrop30%_pretrain', help='path for the pretrained weights')
parser.add_argument('--weight_path', type=str, default='weight/notpretrain/Ours_1yearDrop30%_notpretrain', help='path for saving the weights')
parser.add_argument('--status', type=str, default='train', help='train / test')

args = parser.parse_args()


# Import functions based on whether pretraining is required
if(args.pretrain):
    from exp.MAMTSC_trainer_unsupv import *
    from data_provider.dataset_unsupv import *
else:
    from exp.MAMTSC_trainer_supv import *
    from data_provider.dataset_supv import *    


# Setting new dataset and dataloader
with open(f'dataset/preprocessed_data/training_{args.training_type}_{args.horizon}yearDrop{int(100*args.drop_ratio)}%.pickle','rb') as f:
    train_data = pickle.load(f)

with open(f'dataset/preprocessed_data/validation_balance_{args.horizon}yearDrop{int(100*args.drop_ratio)}%.pickle','rb') as f:
    validate_data = pickle.load(f)

if(args.testing_type == 'custom'):
    with open(f'{args.testing_path}.pickle','rb') as f:
        test_data = pickle.load(f)
else:
    with open(f'dataset/preprocessed_data/testing_{args.testing_type}_{args.horizon}yearDrop{int(100*args.drop_ratio)}%.pickle','rb') as f:
        test_data = pickle.load(f)


base = train_data[0]
train_lab_norm, valid_lab_norm, test_lab_norm = lab_Norm(args.norm, args.seq_len , base, train_data[0], validate_data[0], test_data[0])  # Normalize the laboratory data
train_patient_norm, valid_patient_norm, test_patient_norm = patient_Norm(args.norm, train_data[1], validate_data[1], test_data[1])  # Normalize the patient data

train_label, valid_label, test_label = list(zip(train_data[3], train_data[4], train_data[2])), list(zip(validate_data[3], validate_data[4], validate_data[2])), list(zip(test_data[3], test_data[4], test_data[2]))
train_dataset, validate_dataset, test_dataset = Dataset(list(zip(train_lab_norm, train_patient_norm)), train_label), Dataset(list(zip(valid_lab_norm, valid_patient_norm)), valid_label), Dataset(list(zip(test_lab_norm, test_patient_norm)), test_label)


print("Creating Dataloader...")
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda batch: train_dataset.collate_fn(batch, args.norm), shuffle=True)
validate_data_loader = DataLoader(validate_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda batch: validate_dataset.collate_fn(batch, args.norm), shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=lambda batch: test_dataset.collate_fn(batch, args.norm), shuffle=False)

print(f'number of training data: {len(train_data[0])}')
print(f'number of validation data: {len(validate_data[0])}')
print(f'number of testing data: {len(test_data[0])}')
print()



# Define model
from models.MAMTSC import *
from models.LSTM import *
from models.TCN import *
if(args.model == 'MAMTSC'):
    model = MAMTSC(args)
elif(args.model == 'LSTM'):
    model = LSTM(args)
else:
    model = TCN(args)

    
# Check if model requires finetuning
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

if(args.finetune):
    print('Require finetuning.....')
    if(args.linearprob):
        set_parameter_requires_grad(model, True)
    checkpoint = torch.load(f'{args.pretrain_path}.pth')
    state_dict = deepcopy(checkpoint)

    for key, val in checkpoint.items():
        if key.startswith('output_layer'):
            state_dict.pop(key)

    model.load_state_dict(state_dict, strict=False)

    
# Training and testing
if(args.status == 'train'):
    print('Training.....')
    trainer = MAMTSC_trainer(model, args.weight_path, train_dataloader=train_data_loader, validate_dataloader=validate_data_loader, test_dataloader=None, with_cuda=args.use_gpu, lr=args.learning_rate)

    for epoch in range(args.epochs):
        print("epoch: " + str(epoch + 1) + ' / ' + str(args.epochs))

        trainer.train(epoch)

        # Validation
        if validate_data_loader is not None:
            trainer.validate(epoch)

    
    trainer.plot_figure()


print('Testing.....')
if(args.model == 'MAMTSC'):
    test_model = MAMTSC(args)
elif(args.model == 'LSTM'):
    test_model = LSTM(args)
else:
    test_model = TCN(args)

test_model.load_state_dict(torch.load(f'{args.weight_path}.pth'))
test_model.eval()

tester = MAMTSC_trainer(test_model, args.weight_path, train_dataloader=None, validate_dataloader=None, test_dataloader=test_data_loader, with_cuda=args.use_gpu, lr=args.learning_rate)

for epoch in range(1):
    test_result = tester.test(epoch)
