import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

        


class LSTM(nn.Module):

    def __init__(self, configs):
        super(LSTM, self).__init__()
        
        self.feat_dim = configs.enc_in
        self.d_model = configs.d_model
        self.lstm_layers = configs.lstm_layers
        self.lstm_drop_p = configs.lstm_drop_p
        
        self.lstm = nn.LSTM(input_size = self.feat_dim, hidden_size = self.d_model, num_layers = self.lstm_layers, batch_first=True, dropout=self.lstm_drop_p)
        self.output_layer = nn.Linear(self.d_model, 1)


    
    def forward(self, lab_input, PT_input, freq_input):

        lab_timeSeries_feat, (hn, cn) = self.lstm(lab_input)


        output = F.sigmoid(self.output_layer(lab_timeSeries_feat[:,-1,:]))
        
        return output

