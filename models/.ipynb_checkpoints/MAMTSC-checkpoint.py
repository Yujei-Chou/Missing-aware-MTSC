import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Linear, Dropout, TransformerEncoderLayer
from utils.TransEncoder import TransformerBatchNormEncoderLayer, get_pos_encoder, _get_activation_fn
import numpy as np
import math


class MAMTSC(nn.Module):

    def __init__(self, configs):
        super(MAMTSC, self).__init__()

        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.d_feedfoward = configs.d_feedforawd
        self.n_heads = configs.n_heads
        self.num_layers = configs.e_layers
        self.feat_dim = configs.enc_in
        self.dropout = configs.drop_p
        self.if_FM = configs.if_FM
        self.pretrain = configs.pretrain
        self.pos_encoding = 'fixed'
        self.activation = 'gelu'
        self.norm = 'BatchNorm'
        
        
        if(self.if_FM): # if employing Fusion Module
            self.lab_freq_emb = nn.Sequential(nn.Linear(2, self.d_model),
                                              nn.ReLU(),
                                              nn.Dropout(0.1),
                                              nn.Linear(self.d_model, 1))
        
        
        self.project_inp = nn.Linear(self.feat_dim, self.d_model)
        self.pos_enc = get_pos_encoder(self.pos_encoding)(self.d_model, dropout=self.dropout, max_len=self.seq_len)

        if(self.norm == 'LayerNorm'):
            encoder_layer = TransformerEncoderLayer(self.d_model, self.n_heads, self.d_feedfoward, self.dropout, activation=self.activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, self.d_feedfoward, self.dropout, activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.act = _get_activation_fn(self.activation)

        self.dropout1 = nn.Dropout(self.dropout)

        if(self.pretrain):
            self.output_layer = nn.Linear(self.d_model, self.feat_dim)
        else:
            self.output_layer = nn.Linear(self.d_model*self.seq_len, 1)
        
        

    def forward(self, lab_input, PT_input, freq_input):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        
        if(self.if_FM): # if employing Fusion Module 
            TF_input = torch.cat((lab_input.unsqueeze(-1), freq_input.unsqueeze(-1)), dim = -1)
            inp = self.lab_freq_emb(TF_input).squeeze(-1).permute(1, 0, 2)
        else:
            inp = lab_input.permute(1, 0, 2)
        
        inp = self.project_inp(inp) * math.sqrt(self.d_model)  # (seq_length, batch_size, d_model)
        inp = self.pos_enc(inp)                                # add positional encoding
        output = self.transformer_encoder(inp)                 # embedding encoded with positional and missing information passed through Transformer Encoder
        output = self.act(output)
        output = output.permute(1, 0, 2)
        output = self.dropout1(output)
        
        # Output
        if(self.pretrain):
            output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        else:
            output = F.sigmoid(self.output_layer(output.reshape(output.size()[0], -1)))  # (batch_size, 1)

        return output
