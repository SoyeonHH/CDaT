"""
reference: Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." arXiv preprint arXiv:1707.07250 (2017). https://github.com/Justin1904/TensorFusionNetworks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from .until_module import getBinaryTensor, GradReverse, CTCModule, get_kt_loss


class EmotionClassifier(nn.Module):
    def __init__(self, input_dims, num_classes, dropout=0.1):
        super(EmotionClassifier, self).__init__()
        self.dense = nn.Linear(input_dims, num_classes)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq_input):
        out = self.dense(seq_input)
        out = self.dropout(out)
        out = self.activation(out)
        return out


class SubNet(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(SubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1


class Early(nn.Module):
    
    def __init__(self, config, device, kt_loss_weight=None):
        super(Early, self).__init__()

        # Configuration
        self.config = config
        self.device = device
        self.num_classes = config.num_classes
        self.aligned = config.aligned
        self.kt_loss_weight = kt_loss_weight
        self.threshold = config.threshold
        
        self.text_in = config.text_dim
        self.video_in = config.video_dim
        self.audio_in = config.audio_dim

        # self.text_hidden = hidden_dims[0]
        # self.video_hidden = hidden_dims[1]
        # self.audio_hidden = hidden_dims[2]

        # self.text_out = text_out
        # self.post_fusion_dim = post_fusion_dim

        # self.text_dropout = dropouts[0] 
        # self.video_dropout = dropouts[1]
        # self.audio_dropout = dropouts[2]
        # self.post_fusion_dropout = dropouts[3]

        # # define the pre-fusion subnetworks
        # self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_hidden, dropout=self.audio_dropout)
        # self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_hidden, dropout=self.video_dropout)
        # self.text_subnet = SubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_dropout)
        
        # define the classifier
        self.classifier = EmotionClassifier(self.text_in + self.video_in + self.audio_in, config.num_classes)
        self.ml_loss = nn.BCELoss()
        
        if self.aligned == False:
            self.a2t_ctc = CTCModule(config.audio_dim, 50)
            self.v2t_ctc = CTCModule(config.video_dim, 50)
            self.ctc_criterion = CTCLoss()  


    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask, 
                label_input, label_mask, groundTruth_labels=None, training=True, kt_training=False, dynamic_weight=None):
        '''
        text: [B, L, Dt]
        visual: [B, L, Dv]
        audio: [B, L, Da]
        '''

        batch_size = groundTruth_labels.size(0)
        
        # extract features from subnets
        text = text.view(batch_size, -1, self.text_in).type(torch.FloatTensor).to(self.device)
        visual = visual.view(batch_size, -1, self.video_in).type(torch.FloatTensor).to(self.device)
        audio = audio.view(batch_size, -1, self.audio_in).type(torch.FloatTensor).to(self.device)
        
        averaged_t = torch.mean(text, dim=1)
        averaged_v = torch.mean(visual, dim=1)
        averaged_a = torch.mean(audio, dim=1)

        # text_h = self.text_subnet(text)
        # video_h = self.video_subnet(visual)
        # audio_h = self.audio_subnet(audio)

        # # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        # if audio_h.is_cuda:
        #     DTYPE = torch.cuda.FloatTensor
        # else:
        #     DTYPE = torch.FloatTensor
            
        concat_tensor = torch.cat((averaged_t, averaged_v, averaged_a), dim=1).type(torch.FloatTensor).to(self.device)
        concat_tensor = concat_tensor.view(batch_size, -1)

        # apply the classifier
        predicted_scores = self.classifier(concat_tensor)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, boundary=self.threshold)
        groundTruth_labels = groundTruth_labels.view(-1, self.num_classes)

        # predicted_score = predicted_scores.flatten()
        # labels = labels.flatten()

        # loss
        if training:
            cls_loss = self.ml_loss(predicted_scores, groundTruth_labels)

            if self.kt_loss_weight is not None and self.config.use_kt:
                kt_loss = get_kt_loss(averaged_t, averaged_v, averaged_a, groundTruth_labels, dynamic_weight=dynamic_weight)
                loss = cls_loss + self.kt_loss_weight * kt_loss
            else:
                loss = cls_loss
        
            return loss, predicted_labels, groundTruth_labels, predicted_scores
        else:
            ml_loss = self.ml_loss(predicted_scores, groundTruth_labels)
            return ml_loss, predicted_labels, groundTruth_labels, predicted_scores
        
    
    def inference(self, text, text_mask, visual, visual_mask, audio, audio_mask, \
                label_input, label_mask, groundTruth_labels=None, masked_modality=None):
        
        label_input = label_input.unsqueeze(0)
        batch = groundTruth_labels.size(0)
        label_input = label_input.repeat(batch, 1)
        label_mask = label_mask.unsqueeze(0).repeat(batch, 1)
        
        text = text.view(batch, -1, self.text_in).type(torch.FloatTensor).to(self.device)
        visual = visual.view(batch, -1, self.video_in).type(torch.FloatTensor).to(self.device)
        audio = audio.view(batch, -1, self.audio_in).type(torch.FloatTensor).to(self.device)
        
        averaged_t = torch.mean(text, dim=1)
        averaged_v = torch.mean(visual, dim=1)
        averaged_a = torch.mean(audio, dim=1)

        # text_h = self.text_subnet(text)
        # video_h = self.video_subnet(visual)
        # audio_h = self.audio_subnet(audio)
        
        # # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        # if audio_h.is_cuda:
        #     DTYPE = torch.cuda.FloatTensor
        # else:
        #     DTYPE = torch.FloatTensor
        
        # Modalilty masking before fusion with zero padding
        if masked_modality is not None:
            if "text" in masked_modality:
                averaged_t = torch.zeros_like(averaged_t)
            if "video" in masked_modality:
                averaged_v = torch.zeros_like(averaged_v)
            if "audio" in masked_modality:
                averaged_a = torch.zeros_like(averaged_a)

        concat_tensor = torch.cat((averaged_t, averaged_v, averaged_a), dim=1).type(torch.FloatTensor).to(self.device)
        concat_tensor = concat_tensor.view(batch, -1)
        
        predicted_scores = self.classifier(concat_tensor)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores)
        groundTruth_labels = groundTruth_labels.view(-1, self.num_classes)
        
        return predicted_scores, predicted_labels, concat_tensor, groundTruth_labels
        