import numpy as np
import random
import logging
from numpy import exp
import os

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from transformers import BertModel, BertConfig, BertTokenizer

from .until_module import *
from .module_bert import BertModel, BertConfig, BertOnlyMLMHead

import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

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
    

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)
    
def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class MISA(nn.Module):
    """MISA model for CMU-MOSEI emotion multi-label classification"""
    def __init__(self, config, device, kt_loss_weight=None):
        super(MISA, self).__init__()

        self.config = config
        self.device = device
        self.kt_loss_weight = kt_loss_weight
        self.threshold = config.threshold
        
        self.text_size = config.text_dim
        self.visual_size = config.video_dim
        self.acoustic_size = config.audio_dim

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = 0.5
        self.activation = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.ml_loss = nn.BCELoss()

        ## Initialize the model
        rnn = nn.LSTM

        self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))



        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        
        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        
        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        # if not self.config.use_cmd_sim:
        #     self.discriminator = nn.Sequential()
        #     self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        #     self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
        #     self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
        #     self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))

        self.classifier = EmotionClassifier(config.hidden_size*6, num_classes=output_size)
        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)


    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, enforce_sorted=False, batch_first=True).to(dtype=torch.float32)
        # packed_sequence = packed_sequence[0]

        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, enforce_sorted=False)

        _, (final_h2, _) = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask, 
                label_input, label_mask, groundTruth_labels=None, training=True, kt_training=False, dynamic_weight=None):
        """
        text: [B, L, Dt]
        visual: [B, L, Dv]
        audio: [B, L, Da]
        """
        
        label_input = label_input.unsqueeze(0)
        batch_size = text.size(0)
        lengths = torch.LongTensor([sample.size(0) for sample in text])

        # extract features from text modality
        
        final_h1t, final_h2t = self.extract_features(text, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(audio, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)


        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        self.domain_label_t = None
        self.domain_label_v = None
        self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)

        predicted_scores = self.classifier(h)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, boundary=self.threshold)
        groundTruth_labels = groundTruth_labels.view(-1, self.config.num_classes)

        if training:
            
            all_loss = 0.
            ml_loss = self.ml_loss(predicted_scores, groundTruth_labels)
            # domain_loss = get_domain_loss(self.config, self.domain_label_t, self.domain_label_v, self.domain_label_a)
            cmd_loss = get_cmd_loss(self.config, self.utt_shared_t, self.utt_shared_v, self.utt_shared_a)
            diff_loss = get_diff_loss([self.utt_shared_t, self.utt_shared_v, self.utt_shared_a], [self.utt_private_t, self.utt_private_v, self.utt_private_a])
            recon_loss = get_recon_loss([self.utt_t_recon, self.utt_v_recon, self.utt_a_recon], [self.utt_t_orig, self.utt_v_orig, self.utt_a_orig])
            
            # loss
            # if self.config.use_cmd_sim:
            similarity_loss = cmd_loss
            # else:
            # similarity_loss = domain_loss

            all_loss = ml_loss + 0.3 * diff_loss + 0.7 * similarity_loss + 0.7 * recon_loss

            if self.kt_loss_weight is not None and self.config.use_kt:
                kt_loss = get_kt_loss(self.utt_shared_t, self.utt_shared_v, self.utt_shared_a, groundTruth_labels, dynamic_weight=dynamic_weight)
                all_loss += self.kt_loss_weight * kt_loss
            return all_loss, predicted_labels, groundTruth_labels, predicted_scores
        else:
            ml_loss = self.ml_loss(predicted_scores, groundTruth_labels)
            return ml_loss, predicted_labels, groundTruth_labels, predicted_scores

    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)        
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def inference(self, text, text_mask, visual, visual_mask, audio, audio_mask, \
                label_input, label_mask, groundTruth_labels=None, masked_modality=None):
        label_input = label_input.unsqueeze(0)
        batch_size = text.size(0)
        lengths = torch.LongTensor([sample.size(0) for sample in text])
        
        final_h1t, final_h2t = self.extract_features(text, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        final_h1a, final_h2a = self.extract_features(audio, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        self.domain_label_t = None
        self.domain_label_v = None
        self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()

        # Modalilty masking before fusion with zero padding
        if masked_modality is not None:
            if "text" in masked_modality:
                # self.utt_private_t = torch.zeros_like(self.utt_private_t)
                self.utt_shared_t = torch.zeros_like(self.utt_shared_t)
            if "video" in masked_modality:
                # self.utt_private_v = torch.zeros_like(self.utt_private_v)
                self.utt_shared_v = torch.zeros_like(self.utt_shared_v)
            if "audio" in masked_modality:
                # self.utt_private_a = torch.zeros_like(self.utt_private_a)
                self.utt_shared_a = torch.zeros_like(self.utt_shared_a)
        
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)

        predicted_scores = self.classifier(h)
        predicted_scores = predicted_scores.view(-1, self.config.num_classes)
        predicted_labels = getBinaryTensor(predicted_scores, boundary=self.threshold)
        groundTruth_labels = groundTruth_labels.view(-1, self.config.num_classes)
        
        return predicted_scores, predicted_labels, h, groundTruth_labels