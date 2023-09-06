import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class ConfidenceRegressionNetwork(nn.Module):
    def __init__(self, config, input_dims, num_classes=1, dropout=0.1):
        super(ConfidenceRegressionNetwork, self).__init__()
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))
    
        # self.sigmoid = nn.Sigmoid()
        
        self.loss_tcp = nn.MSELoss(reduction='mean')
    
    def forward(self, seq_input, targets):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        loss = self.loss_tcp(output, targets)

        return loss, output
    
    def inference(self, seq_input):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        
        return output

class Confidnet3Layers(nn.Module):
    def __init__(self, config, input_dims, num_classes=1, dropout=0.1) -> None:
        super(Confidnet3Layers, self).__init__()
        self.config = config
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes))
        
        self.loss_tcp = nn.MSELoss(reduction='mean')
    
    def forward(self, seq_input, targets):
        output = self.mlp(seq_input)
        loss = self.loss_tcp(output, targets)

        return loss, output

    def inference(self, seq_input):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        
        return output

class Confidnet4Layers(nn.Module):
    def __init__(self, config, input_dims, num_classes=1, dropout=0.5):
        super(Confidnet4Layers, self).__init__()
        self.config = config

        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))
    
        # self.sigmoid = nn.Sigmoid()
        
        self.loss_tcp = nn.MSELoss(reduction='mean')
    
    def forward(self, seq_input, targets):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        loss = self.loss_tcp(output, targets)

        return loss, output
    
    def inference(self, seq_input):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        
        return output
    

class Confidnet8Layers(nn.Module):
    def __init__(self, config, input_dims, num_classes=1, dropout=0.1):
        super(Confidnet8Layers, self).__init__()
        self.config = config
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))
            
        self.loss_tcp = nn.MSELoss(reduction='mean')
        
    def forward(self, seq_input, targets):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        loss = self.loss_tcp(output, targets)

        return loss, output
    
    def inference(self, seq_input):
        output = self.mlp(seq_input)
        # output = self.sigmoid(output)
        
        return output