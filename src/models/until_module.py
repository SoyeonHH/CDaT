# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable
import torch.nn.functional as F
import math
from .until_config import PretrainedConfig

logger = logging.getLogger(__name__)

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        self.weight.to(x.device)
        self.bias.to(x.device)
        
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model

##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class MILNCELoss(nn.Module):
    def __init__(self, batch_size=1, n_pair=1,):
        super(MILNCELoss, self).__init__()
        self.batch_size = batch_size
        self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix):
        mm_mask = np.eye(self.batch_size)
        mm_mask = np.kron(mm_mask, np.ones((self.n_pair, self.n_pair)))
        mm_mask = torch.tensor(mm_mask).float().to(sim_matrix.device)

        from_text_matrix = sim_matrix + mm_mask * -1e12
        from_video_matrix = sim_matrix.transpose(1, 0)

        new_sim_matrix = torch.cat([from_video_matrix, from_text_matrix], dim=-1)
        logpt = F.log_softmax(new_sim_matrix, dim=-1)

        mm_mask_logpt = torch.cat([mm_mask, torch.zeros_like(mm_mask)], dim=-1)
        masked_logpt = logpt + (torch.ones_like(mm_mask_logpt) - mm_mask_logpt) * -1e12

        new_logpt = -torch.logsumexp(masked_logpt, dim=-1)

        logpt_choice = torch.zeros_like(new_logpt)
        mark_ind = torch.arange(self.batch_size).to(sim_matrix.device) * self.n_pair + (self.n_pair//2)
        logpt_choice[mark_ind] = 1
        sim_loss = new_logpt.masked_select(logpt_choice.to(dtype=self.bool_dtype)).mean()
        return sim_loss

class MaxMarginRankingLoss(nn.Module):
    def __init__(self,
                 margin=1.0,
                 negative_weighting=False,
                 batch_size=1,
                 n_pair=1,
                 hard_negative_rate=0.5,
        ):
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        self.n_pair = n_pair
        self.batch_size = batch_size
        easy_negative_rate = 1 - hard_negative_rate
        self.easy_negative_rate = easy_negative_rate
        self.negative_weighting = negative_weighting
        if n_pair > 1 and batch_size > 1:
            alpha = easy_negative_rate / ((batch_size - 1) * (1 - easy_negative_rate))
            mm_mask = (1 - alpha) * np.eye(self.batch_size) + alpha
            mm_mask = np.kron(mm_mask, np.ones((n_pair, n_pair)))
            mm_mask = torch.tensor(mm_mask) * (batch_size * (1 - easy_negative_rate))
            self.mm_mask = mm_mask.float()

    def forward(self, x):
        d = torch.diag(x)
        max_margin = F.relu(self.margin + x - d.view(-1, 1)) + \
                     F.relu(self.margin + x - d.view(1, -1))
        if self.negative_weighting and self.n_pair > 1 and self.batch_size > 1:
            max_margin = max_margin * self.mm_mask.to(max_margin.device)
        return max_margin.mean()


def Focalloss(predictions, labels, weights=None, alpha=0.25, gamma=2):


    """Compute focal loss for predictions.
    Multi-labels Focal loss formula:
    FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
            ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
    predictions: A float tensor of shape [batch_size, 
    num_classes] representing the predicted logits for each class
    target_tensor: A float tensor of shape [batch_size,
    num_classes] representing one-hot encoded classification targets
    weights: A float tensor of shape [batch_size]
    alpha: A scalar tensor for focal loss alpha hyper-parameter
    gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
    loss: A (scalar) tensor representing the value of the loss function
    """ 
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.

    zeros = torch.zeros_like(predictions, dtype=predictions.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = torch.where(labels > zeros, labels - predictions, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = torch.where(labels > zeros, zeros, predictions)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(predictions, 1e-8, 1.0)) \
                            - (1 - alpha) * (neg_p_sub ** gamma) * torch.log(torch.clamp(1.0 - predictions, 1e-8, 1.0))
    return torch.mean(torch.sum(per_entry_cross_ent, 1))

def getBinaryTensor(imgTensor, boundary = 0.5):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)

def get_kt_loss(t, v, a, label, dynamic_weight=None, supervised_weights=0):
    '''
    shape of t: (batch_size, hidden_size)
    shape of v: (batch_size, hidden_size)
    shape of a: (batch_size, hidden_size)
    shape of label: (batch_size, num_classes=6)
    '''

    if dynamic_weight is None:
        dynamic_weight = [1, 1, 1, 1, 1, 1]
    else:
        dynamic_weight = dynamic_weight.permute(1, 0)
    
    loss_t_v = torch.mean(dynamic_weight[0] * cosine_similarity_loss(t, v))
    loss_t_a = torch.mean(dynamic_weight[1] * cosine_similarity_loss(t, a))
    
    loss_v_t = torch.mean(dynamic_weight[2] * cosine_similarity_loss(v, t))
    loss_v_a = torch.mean(dynamic_weight[3] * cosine_similarity_loss(v, a))

    loss_a_t = torch.mean(dynamic_weight[4] * cosine_similarity_loss(a, t))
    loss_a_v = torch.mean(dynamic_weight[5] * cosine_similarity_loss(a, v))

    kt_loss = loss_t_v + loss_t_a + loss_v_t + loss_v_a + loss_a_t + loss_a_v

    return kt_loss.squeeze()

def cosine_similarity_loss(source_net, target_net, dim=1, eps=1e-8):
    
    # source_net = source_net.view(source_net.size(0), -1)
    # target_net = target_net.view(target_net.size(0), -1)
    
    source_net = source_net.reshape(source_net.size(0), -1)
    target_net = target_net.reshape(target_net.size(0), -1)
    
    # Normalize each vector by its norm
    source_net_norm = torch.sqrt(torch.sum(source_net**2, dim=dim, keepdim=True))
    source_net = source_net / (source_net_norm + eps)
    source_net[source_net != source_net] = 0    # replace nan with 0

    target_net_norm = torch.sqrt(torch.sum(target_net**2, dim=dim, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate cosine similarity
    source_similarity = torch.mm(source_net, source_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to [0, 1]
    source_similarity = (source_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    source_similarity = source_similarity / torch.sum(source_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate KL divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (source_similarity + eps)))

    return loss


def supervised_loss(source_net, targets, eps=1e-8):
    # TODO: modify labels to be a multi-label vector
    labels = targets.cpu().numpy()
    target_sim = np.zeros((labels.shape[0], labels.shape[0]), dtype='float32')
    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            target_sim[i, j] = 1.0 if labels[i] == labels[j] else 0.0
    
    target_similarity = torch.from_numpy(target_sim).cuda()
    target_similarity = Variable(target_similarity)

    # Normalize each vector by its norm
    source_net_norm = torch.sqrt(torch.sum(source_net**2, dim=1, keepdim=True))
    source_net = source_net / (source_net_norm + eps)
    source_net[source_net != source_net] = 0    # replace nan with 0

    # Calculate cosine similarity
    source_similarity = torch.mm(source_net, source_net.transpose(0, 1))

    # Scale cosine similarity to [0, 1]
    source_similarity = (source_similarity + 1.0) / 2.0

    # Transform them into probabilities
    source_similarity = source_similarity / torch.sum(source_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate KL divergence
    loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (source_similarity + eps)))

    return loss


def distillation_loss(output, target, T):
    """
    Distillation Loss
    :param output:
    :param target:
    :param T:
    :return:
    """
    output = F.log_softmax(output / T)
    target = F.softmax(target / T)
    loss = -torch.sum(target * output) / output.size()[0]
    return loss



class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class CTCModule(nn.Module): #
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        return pseudo_aligned_out, (pred_output_position_inclu_blank)
    
"""
Adapted from https://github.com/declare-lab/MISA/blob/master/src/utils/functions.py
"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None
    
class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1 = torch.nan_to_num(input1)
        input2 = torch.nan_to_num(input2)
        
        input1_mean = torch.mean(input1, dim=0, keepdim=True)
        input2_mean = torch.mean(input2, dim=0, keepdim=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class CMD(nn.Module):

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
    
def get_domain_loss(config, domain_pred_t, domain_pred_v, domain_pred_a):

    criterion = nn.CrossEntropyLoss(reduction="mean")
    
    # if config.use_cmd_sim:
    #     return 0.0

    # True domain labels
    domain_true_t = torch.LongTensor([0]*domain_pred_t.size(0)).to(DEVICE)
    domain_true_v = torch.LongTensor([1]*domain_pred_v.size(0)).to(DEVICE)
    domain_true_a = torch.LongTensor([2]*domain_pred_a.size(0)).to(DEVICE)

    # Stack up predictions and true labels
    domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
    domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

    return criterion(domain_pred, domain_true)

def get_cmd_loss(config, utt_shared_t, utt_shared_v, utt_shared_a):

    loss_cmd = CMD()

    # losses between shared states
    loss = loss_cmd(utt_shared_t, utt_shared_v, 5)
    loss += loss_cmd(utt_shared_t, utt_shared_a, 5)
    loss += loss_cmd(utt_shared_a, utt_shared_v, 5)
    loss = loss/3.0

    return loss

def get_diff_loss(utt_shared, utt_private):

    loss_diff = DiffLoss()

    shared_t = utt_shared[0]
    shared_v = utt_shared[1]
    shared_a = utt_shared[2]
    private_t = utt_private[0]
    private_v = utt_private[1]
    private_a = utt_private[2]

    # Between private and shared
    loss = loss_diff(private_t, shared_t)
    loss += loss_diff(private_v, shared_v)
    loss += loss_diff(private_a, shared_a)

    # Across privates
    loss += loss_diff(private_a, private_t)
    loss += loss_diff(private_a, private_v)
    loss += loss_diff(private_t, private_v)

    return loss

def get_recon_loss(utt_recon, utt_orig):

    # self.loss_recon = MSE()
    loss_recon = nn.MSELoss(reduction="mean")

    loss = loss_recon(utt_recon[0], utt_orig[0])
    loss += loss_recon(utt_recon[1], utt_orig[1])
    loss += loss_recon(utt_recon[2], utt_orig[2])
    loss = loss/3.0
    return loss