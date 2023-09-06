from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
from collections import defaultdict
import csv
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import time
import argparse     
from pathlib import Path
from functools import partial
from datetime import timedelta

from src.models.confidnet import ConfidenceRegressionNetwork, Confidnet3Layers, Confidnet4Layers
from src.models.models_tfn import TFN
from src.models.models_early import Early
from src.models.models_tailor import TAILOR
from src.models.models_misa import MISA
from src.models.optimization import BertAdam
from src.utils.eval import get_metrics
from src.utils.eval_gap import *

from filelock import FileLock
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler, SequentialSampler
from torch.utils.data import random_split
import torch.utils.data as data
from util import parallel_apply, get_logger, get_tcp_target, binary_ce
from src.dataloaders.cmu_dataloader import prep_dataloader
import torch.nn.functional as F
import torch.nn.parallel as parallel
from timm.scheduler.cosine_lr import CosineLRScheduler

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.air.config import RunConfig

ray.init(local_mode=True)

mosei_data_dir = '/data2/multimodal/train_valid_test.pt'
iemocap_data_dir = '/data2/multimodal/IEMOCAP'

global logger
def get_args(description='Multi-modal Multi-label Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default="tailor", type=str, help="one of tailor, tfn, early, misa")
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.") 
    parser.add_argument("--do_test", default=False, action='store_true', help="whether to run test")
    parser.add_argument("--aligned", action='store_true', help="whether train align of unalign dataset")
    parser.add_argument("--data", default="mosei", type=str, help="one of mosei, iemocap")
    parser.add_argument("--data_path", default="/data2/multimodal/train_valid_test.pt", type=str, help='cmu_mosei data_path')
    parser.add_argument("--output_dir", default="/home/soyeon/workspace/Dike/checkpoint", type=str, required=False,
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pretrained_model", default=None, type=str, help="Initial model.")
    parser.add_argument("--confidnet_model", default=None, type=str, help="Initial model.")

    parser.add_argument('--num_thread_reader', type=int, default=0, help='') 
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='upper epoch limit') 
    parser.add_argument('--unaligned_data_path', type=str, default='/amax/cmy/mosei_senti_data_noalign.pkl', help='load unaligned dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay') 
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--text_dim', type=int, default=300, help='text_feature_dimension') 
    parser.add_argument('--video_dim', type=int, default=35, help='video feature dimension')
    parser.add_argument('--audio_dim', type=int, default=74, help='audio_feature_dimension') 
    parser.add_argument('--seed', type=int, default=42, help='random seed') 
    parser.add_argument('--max_words', type=int, default=60, help='')
    parser.add_argument('--max_frames', type=int, default=60, help='')
    parser.add_argument('--max_sequence', type=int, default=60, help='')
    parser.add_argument('--max_label', type=int, default=6, help='')
    parser.add_argument("--bert_model", default="bert-base", type=str, required=False, help="Bert module")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="audio-base", type=str, required=False, help="Audio module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.") 
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") 
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--bert_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=4, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=4, help="Layer No. of audio")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")
    parser.add_argument("--num_classes", default=6, type=int, required=False)
    parser.add_argument("--hidden_size",type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_bert', action='store_true', default=True, help="Changed in the execute process.")
    parser.add_argument('--threshold', default=0.5, type=float, help='the threshold of whether the emotion exists.')
    
     # Train DKT
    parser.add_argument('--use_kt', action='store_true')
    parser.add_argument('--kt_model', type=str, 
                    default='Dynamic-tcp', help='one of {Static, Dynamic-ce, Dynamic-tcp}')
    # parser.add_argument('--kt_weight', type=float, default=10000.0)
    parser.add_argument('--epochs_kt', type=int, default=1)

    # Train ConfidNet
    parser.add_argument('--epochs_conf', type=int, default=500)
    parser.add_argument('--conf_loss', type=str, default='mse', help='one of {mse, focal, ranking}')
    parser.add_argument('--conf_lr', type=float, default=1e-5)
    parser.add_argument('--conf_dropout', type=float, default=0.6)

    args = parser.parse_args()
    # Check paramenters
    if args.gradient_accumulation_steps < 1: 
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    kt_model_name = {'Static': 'const', 'Dynamic-ce': 'ce', 'Dynamic-tcp': 'confidnet'}
    if args.use_kt:
        args.output_dir = os.path.join(args.output_dir, f'{args.data}_{args.model}_{kt_model_name[args.kt_model]}') if args.aligned else \
            os.path.join(args.output_dir, f'{args.data}_unaligned_{args.model}_{kt_model_name[args.kt_model]}')
    else:
        args.output_dir = os.path.join(args.output_dir, f'{args.data}_{args.model}') if args.aligned else \
            os.path.join(args.output_dir, f'{args.data}_unaligned_{args.model}')
            
        
    if args.data == "mosei":
        args.data_path = mosei_data_dir
    elif args.data == "iemocap":
        args.data_path = iemocap_data_dir
        args.num_classes = 4

    return args


def set_seed_logger(args): 
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    torch.cuda.set_device(args.local_rank) 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0: 
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, kt_loss_weight=None): 
        
    # kt_loss_weight = torch.tensor(kt_loss_weight).to(device)

    # Prepare model
    if args.model == "tailor":
        model = TAILOR.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model, args.decoder_model, \
            task_config=args, kt_loss_weight=kt_loss_weight, device=device)
    elif args.model == "tfn":
        model =  TFN(args, (128, 32, 32), 64, (0.3, 0.3, 0.3, 0.3), 128, device, kt_loss_weight=kt_loss_weight)
    elif args.model == "early":
        model = Early(args, device, kt_loss_weight=kt_loss_weight)
    elif args.model == "misa":
        model = MISA(args, device, kt_loss_weight=kt_loss_weight)
    else:
        raise ValueError("Invalid model: {}".format(args.model))
        
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model).to(device)
        
    num_params = count_parameters(model)
    # logger.info("Total Parameter: \t%2.1fM" % num_params)
        
    if hasattr(model, 'module'):
        model = model.module.to(device)

    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * 1.0},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * 1.0},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                        schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                        max_grad_norm=1.0) 
    scheduler = None
    
    return optimizer, scheduler, model


def save_model(args, model, epoch, confidnet=False):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    if confidnet:
        output_model_file = os.path.join(
            args.output_dir, "pytorch_model_confidnet_{}.bin.".format(epoch))
    else:
        output_model_file = os.path.join(
            args.output_dir, "pytorch_model_{}.bin.".format(epoch))
        
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)

    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None, confidnet=False):
    if model_file is None or len(model_file) == 0:
        if confidnet:
            model_file = os.path.join(args.output_dir, "pytorch_model_confidnet_{}.bin.".format(epoch))
        else:
            model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin.".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

        model = TAILOR.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args, device=device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model.to(device)
    else:
        model = None
    return model


def get_dynamic_tcp(model, confidnet, pairs_text, pairs_mask, video, video_mask, audio, audio_mask, label_input, label_mask, ground_label, device):
        
    outputs, _, h, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask,\
        label_input, label_mask, ground_label, dynamic_weight=None)
    
    # Predict the model confidence
    confid_z = confidnet.inference(h)

    # Get the tcp for the masked modalities
    _, _, h_t_removed, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
        label_input, label_mask, ground_label, masked_modality=["text"], dynamic_weight=None)
    tcp_t_removed = confidnet.inference(h_t_removed)
    
    _, _, h_v_removed, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
        label_input, label_mask, ground_label, masked_modality=["video"], dynamic_weight=None)
    tcp_v_removed = confidnet.inference(h_v_removed)
    
    _, _, h_a_removed, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
        label_input, label_mask, ground_label, masked_modality=["audio"], dynamic_weight=None)
    tcp_a_removed = confidnet.inference(h_a_removed)
    
    w_misaligned = [tcp_t_removed, tcp_v_removed, tcp_a_removed]
    
    dynamic_weight = [
        [tcp_t_removed[i] if tcp_t_removed[i] > tcp_v_removed[i] else 0 for i in range(len(tcp_t_removed))],    # text > video
        [tcp_t_removed[i] if tcp_t_removed[i] > tcp_a_removed[i] else 0 for i in range(len(tcp_t_removed))],    # text > audio
        [tcp_v_removed[i] if tcp_v_removed[i] > tcp_t_removed[i] else 0 for i in range(len(tcp_v_removed))],    # video > text
        [tcp_v_removed[i] if tcp_v_removed[i] > tcp_a_removed[i] else 0 for i in range(len(tcp_v_removed))],    # video > audio
        [tcp_a_removed[i] if tcp_a_removed[i] > tcp_t_removed[i] else 0 for i in range(len(tcp_a_removed))],    # audio > text
        [tcp_a_removed[i] if tcp_a_removed[i] > tcp_v_removed[i] else 0 for i in range(len(tcp_a_removed))]    # audio > video
    ]

    dynamic_weight = torch.tensor(dynamic_weight, dtype=torch.float).permute(1, 0).to(device)
    return dynamic_weight, w_misaligned


def get_dynamic_ce(model, confidnet, pairs_text, pairs_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_label, device):
    
    output, _, _, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask,\
        label_input, label_mask, ground_label, dynamic_weight=None)

    output_t_removed, _, _, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
        label_input, label_mask, ground_label, masked_modality=["text"], dynamic_weight=None)
    output_v_removed, _, _, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
        label_input, label_mask, ground_label, masked_modality=["video"], dynamic_weight=None)
    output_a_removed, _, _, _, _ = model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
        label_input, label_mask, ground_label, masked_modality=["audio"], dynamic_weight=None)
    
    t_mask_loss = binary_ce(output, output_t_removed)
    v_mask_loss = binary_ce(output, output_v_removed)
    a_mask_loss = binary_ce(output, output_a_removed)
    
    w_misaligned = [t_mask_loss, v_mask_loss, a_mask_loss]
    
    dynamic_weight = [
        [t_mask_loss[i] if t_mask_loss[i] > v_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
        [t_mask_loss[i] if t_mask_loss[i] > a_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
        [v_mask_loss[i] if v_mask_loss[i] > t_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
        [v_mask_loss[i] if v_mask_loss[i] > a_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
        [a_mask_loss[i] if a_mask_loss[i] > t_mask_loss[i] else 0 for i in range(len(t_mask_loss))], \
        [a_mask_loss[i] if a_mask_loss[i] > v_mask_loss[i] else 0 for i in range(len(t_mask_loss))]
    ]
    
    dynamic_weight = torch.tensor(dynamic_weight, dtype=torch.float).to(device)

    return dynamic_weight, w_misaligned


def train(args, device, n_gpu, n_epochs=40):
    global logger
    
    # init model
    model = init_model(args, device)
    
    if args.aligned == False:
        logger.warning("!!!!!!!!!!!!!! you start train unaligned dataset")
    else:
        logger.warning("!!!!!!!!!!!!!! you start train aligned dataset")
    print('***** dataloder preping ... *****')

    
    train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask = prep_dataloader(args)
    label_input = label_input.to(device)
    label_mask = label_mask.to(device)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    coef_lr = args.coef_lr
    if args.init_model:
        coef_lr = 1.0
        
    # init optimizer
    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, \
        args.local_rank, coef_lr=coef_lr)
    
    # if args.local_rank == 0:
    logger.info("***** Running baseline training *****")
    logger.info("  Total optimization epochs = %d", n_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size / args.n_gpu)

    best_score = 0.000
    best_output_model_file = None
    global_step = 0
    best_model = None
    model.zero_grad()
    set_seed_logger(args)   # Added here for reproductibility
    
    for epoch in range(n_epochs):    # loop over the dataset multiple times
        model.train()
        log_step = args.n_display
        local_rank = args.local_rank
        start_time = time.time()
        total_loss = 0
        total_pred = []
        total_true_label = []
        total_pred_scores = [] 
        
        for step, batch in enumerate(train_dataloader):
        #   torch.cuda.empty_cache()
            if n_gpu == 1:
                # multi-gpu does scattering it-self
                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

            pairs_text, pairs_mask, video, video_mask,audio, audio_mask, ground_label = batch
            dynamic_weight = None
            
            model_loss, batch_pred, true_label, pred_scores = model(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
                    groundTruth_labels=ground_label, training=True, kt_training=False, dynamic_weight=dynamic_weight)

            if n_gpu > 1:
                model_loss = model_loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                model_loss = model_loss / args.gradient_accumulation_steps
            model_loss.backward() 
            
            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

                if scheduler is not None:
                    scheduler.step(epoch)  # Update learning rate schedule

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                # if global_step % log_step == 0 and local_rank == 0:
                if global_step % log_step == 0:
                    logger.info("Epoch: %d/%d, Step: %d/%d, Lr: %s, loss: %f,  Time/step: %f", epoch + 1,
                                args.epochs, step + 1,
                                # len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),float(model_loss), 
                                len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set([param_group['lr'] for param_group in optimizer.param_groups])))]),float(model_loss),
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                    start_time = time.time()

            total_loss += float(model_loss)
            total_pred.append(batch_pred)
            total_true_label.append(true_label)
            total_pred_scores.append(pred_scores)
        
        if scheduler is not None:
            scheduler.step(epoch)
        
        total_loss = total_loss / len(train_dataloader)
        total_pred=torch.cat(total_pred,0)
        total_true_label = torch.cat(total_true_label, 0)
        total_pred_scores = torch.cat(total_pred_scores, 0)
        
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_true_label)
        total_pred_scores = total_pred_scores.data.cpu().numpy()
        total_true_label = total_true_label.data.cpu().numpy()
        train_gap = calculate_gap(total_pred_scores, total_true_label)
        
        # if args.local_rank == 0:
        logger.info("Epoch %d/%d Finished, Train Loss: %f, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f, train_gap: %f",  \
            epoch + 1, args.epochs, total_loss, total_micro_f1, total_micro_precision, total_micro_recall,  total_acc, train_gap)
        
        # if args.local_rank == 0:
        # Validation
        logger.info("***** Running baseline valing *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch_size = %d", args.batch_size)
        # val_pred, val_label, val_pred_scores, val_loss = eval_epoch(args, model, val_dataloader, device, n_gpu, label_input, label_mask)
        
        if hasattr(model, 'module'):
            model = model.module.to(device)
        # else:            
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model).to(device)
        # else:
        #     model.to(device)

        model.eval()
        with torch.no_grad():
            val_pred = []
            val_label = []
            val_pred_scores = []
            losses = []
            for _, batch in enumerate(val_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
                _, batch_pred, true_label, pred_scores = model(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, groundTruth_labels=groundTruth_labels, training=False)
                val_pred.append(batch_pred)
                val_label.append(true_label)
                val_pred_scores.append(pred_scores)
            
            val_pred=torch.cat(val_pred,0)
            val_label = torch.cat(val_label, 0)
            val_pred_scores = torch.cat(val_pred_scores, 0)
                
            val_micro_f1, val_micro_precision, val_micro_recall, val_acc = get_metrics(val_pred, val_label)
            val_pred_scores = val_pred_scores.data.cpu().numpy()
            val_label = val_label.data.cpu().numpy()
            val_gap = calculate_gap(val_pred_scores, val_label)   
            
            logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f, val_gap: %f", \
                val_micro_f1, val_micro_precision, val_micro_recall, val_acc, val_gap)
            output_model_file = save_model(args, model, epoch)
            
            if best_score <=  val_micro_f1:
                best_score = val_micro_f1
                best_model = model
                
                best_output_model_file = output_model_file
            logger.info("The best model is: {}, the f1 is: {:.4f}".format(best_output_model_file, best_score))
    logger.info("Training finished!")
    
    logger.info("***** Running baseline testing *****")
    logger.info("  Num examples = %d", test_length)
    logger.info("  Batch_size = %d", args.batch_size)
    
    if hasattr(best_model, 'module'):
        best_model = best_model.module.to(device)
    # else:
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model).to(device)
    # else:
    #     model.to(device)

    best_model.eval()
    
    with torch.no_grad():
        test_pred = []
        test_label = []
        test_pred_scores = []
        losses = []
        for _, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
            _, batch_pred, true_label, pred_scores = best_model(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, groundTruth_labels=groundTruth_labels, training=False)
            test_pred.append(batch_pred)
            test_label.append(true_label)
            test_pred_scores.append(pred_scores)
        
        test_pred=torch.cat(test_pred,0)
        test_label = torch.cat(test_label, 0)
        test_pred_scores = torch.cat(test_pred_scores, 0)
        
    test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(test_pred, test_label)
    test_pred_scores = test_pred_scores.data.cpu().numpy()
    test_label = test_label.data.cpu().numpy()
    test_gap = calculate_gap(test_pred_scores, test_label)
    
    logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f, test_gap: %f", \
        test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_gap)
        
    
    return best_model, best_output_model_file
    

def train_confidnet(args, model, device, n_gpu, n_epochs=100):
    global logger
    local_rank = args.local_rank
    
    assert model is not None, "Please specify the exact model !"
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model).to(device)
    # else:
    #     model.to(device)
    model.eval()
    
    # init confidence network
    if args.model == "tailor":
        hidden_size = args.hidden_size * args.num_classes
    elif args.model == "tfn":
        hidden_size = 128
    elif args.model == "early":
        hidden_size = args.text_dim + args.video_dim + args.audio_dim
    elif args.model == "misa":
        hidden_size = args.hidden_size * 6

    confidnet = Confidnet4Layers(args, hidden_size)
    
    if args.confidnet_model is not None:
        confidnet.load_state_dict(torch.load(args.confidnet_model))
        return confidnet
    
    confidnet = confidnet.to(device)
    conf_optimizer = torch.optim.Adam(confidnet.parameters(), lr=args.conf_lr)
    
    train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, \
        label_input, label_mask = prep_dataloader(args, zero_label_process=True)
    label_input = label_input.to(device)
    label_mask = label_mask.to(device)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                    / args.gradient_accumulation_steps) * args.epochs
    
    logger.info("***** Running ConfidNet training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
    
    best_score = 1e+10
    best_output_model_file = None
    global_step = 0
    for epoch in range(n_epochs):
        confidnet.train()
        log_step = args.n_display
        start_time = time.time()
        total_loss = 0
        
        for param in model.parameters():
            param.requires_grad = False
            
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            pairs_text, pairs_mask, video, video_mask,audio, audio_mask, ground_label = batch
            
            batch_pred, pred_labels, hidden_state, true_labels, _ = \
                model.inference(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, \
                    label_input, label_mask, groundTruth_labels=ground_label)
            target_tcp = get_tcp_target(ground_label, batch_pred)
            loss, preds = confidnet(hidden_state, target_tcp)
            loss.backward()
            total_loss += float(loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                torch.nn.utils.clip_grad_norm_(confidnet.parameters(), 1.0) 
                
                conf_optimizer.step()
                conf_optimizer.zero_grad()
                
                global_step += 1
                # if global_step % log_step == 0 and local_rank == 0:
                if global_step % log_step == 0:
                    logger.info("Epoch: %d/%d, Step: %d/%d, loss: %f,  Time/step: %f", epoch + 1,
                                        args.epochs, step + 1,
                                        len(train_dataloader), float(loss),
                                        (time.time() - start_time))
                    start_time = time.time()
            
        total_loss = total_loss / len(train_dataloader)

        # if args.local_rank == 0:
        logger.info("Epoch %d/%d Finished, Train Loss: %f",  \
            epoch + 1, n_epochs, total_loss)
        # if args.local_rank == 0:
        logger.info("***** Running ConfidNet valing *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch_size = %d", args.batch_size)
        
        # Validation
        # val_loss = conf_eval_epoch(args, model, confidnet, val_dataloader, device, label_input, label_mask)
        confidnet.eval()

        for param in model.parameters():
            param.requires_grad = False
        
        with torch.no_grad():
            val_loss = 0
            for _, batch in enumerate(val_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
                batch_pred, pred_labels, hidden_state, true_labels, _ = \
                    model.inference(text, text_mask, video, video_mask, audio, audio_mask, \
                        label_input, label_mask, groundTruth_labels=groundTruth_labels)
                target_tcp = get_tcp_target(groundTruth_labels, batch_pred)
                loss, preds = confidnet(hidden_state, target_tcp)
                val_loss += float(loss)
            
            val_loss = total_loss / len(val_dataloader)
            
            logger.info("----- val_loss: %f", val_loss)
            
            output_model_file = save_model(args, confidnet, epoch, confidnet=True)
            if best_score >= val_loss:
                best_score = val_loss
                best_confidnet = confidnet
                best_output_model_file = output_model_file
            logger.info("The best confidnet is: {}, the loss is: {:.4f}".format(best_output_model_file, best_score))
    
    # if args.local_rank == 0:
    logger.info('***** Running ConfidNet testing *****')
    logger.info('  Num examples = %d', test_length)
    logger.info("  Batch_size = %d", args.batch_size)
        
    # Test
    # test_loss = conf_eval_epoch(args, model, best_confidnet, test_dataloader, device, label_input, label_mask)
    confidnet.eval()

    for param in model.parameters():
        param.requires_grad = False
    
    with torch.no_grad():
        test_loss = 0
        for _, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
            batch_pred, pred_labels, hidden_state, true_labels, _ = \
                model.inference(text, text_mask, video, video_mask, audio, audio_mask, \
                    label_input, label_mask, groundTruth_labels=groundTruth_labels)
            target_tcp = get_tcp_target(groundTruth_labels, batch_pred)
            loss, preds = confidnet(hidden_state, target_tcp)
            test_loss += float(loss)
        
        test_loss = total_loss / len(test_dataloader)
        logger.info("----- test_loss: %f", test_loss)
    
    return confidnet
    

def train_kt(tune_config, args, device, n_gpu, pretrained_model=None, confidnet=None, n_epochs=50):
# def train_kt(args, device, n_gpu, pretrained_model=None, confidnet=None, n_epochs=50):
    # @ray.remote
    global logger
    train_time = time.time()
        
    # init model
    model = init_model(args, device, tune_config["kt_loss_weight"]) 
    # model = init_model(args, device, kt_loss_weight=1)

    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model))
    
    if args.aligned == False:
        logger.warning("!!!!!!!!!!!!!! you start train unaligned dataset")
    else:
        logger.warning("!!!!!!!!!!!!!! you start train aligned dataset")
    print('***** dataloder preping ... *****')

    
    train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask = prep_dataloader(args)
    label_input = label_input.to(device)
    label_mask = label_mask.to(device)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs

    coef_lr = args.coef_lr
    if args.init_model:
        coef_lr = 1.0
        
    # init optimizer
    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, \
        args.local_rank, coef_lr=coef_lr)

    # To restore a checkpoint, use `session.get_checkpoint()`.
    # loaded_checkpoint = session.get_checkpoint()
    # if loaded_checkpoint:
    #     with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #        model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
    #     model.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    # if args.local_rank == 0:
    logger.info("***** Running Dike training *****")
    logger.info("  Num examples = %d", train_length)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

    best_score = 0.000
    best_output_model_file = None
    global_step = 0
    best_model = None
    
    for epoch in range(n_epochs):    # loop over the dataset multiple times
                
        model.train()
        log_step = args.n_display
        local_rank = args.local_rank
        start_time = time.time()
        total_loss = 0
        total_pred = []
        total_true_label = []
        total_pred_scores = [] 
        w_misaligned_dict = defaultdict(list)
        
        # breakpoint()
        
        for step, batch in enumerate(train_dataloader):
        #   torch.cuda.empty_cache()
            if n_gpu == 1:
                # multi-gpu does scattering it-self
                batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

            pairs_text, pairs_mask, video, video_mask,audio, audio_mask, ground_label = batch
            
            w = None
            
            # get dynamic weight
            if args.kt_model == "Dynamic-tcp" and confidnet is not None:
                dynamic_weight, w = get_dynamic_tcp(model, confidnet, pairs_text, pairs_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_label, device)
            elif args.kt_model == "Dynamic-ce":
                dynamic_weight, w = get_dynamic_ce(model, confidnet, pairs_text, pairs_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_label, device)
            else:
                dynamic_weight = None
                
            # if kt_train and epoch in [args.epochs_kt - 1, args.epochs_kt - 1-10, args.epochs_kt - 1-20, 0]:
            #     w_misaligned_dict['t_mask'].extend(w[0].cpu().detach().numpy())
            #     w_misaligned_dict['v_mask'].extend(w[1].cpu().detach().numpy())
            #     w_misaligned_dict['a_mask'].extend(w[2].cpu().detach().numpy())
                
            model_loss, batch_pred, true_label, pred_scores = model(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
                groundTruth_labels=ground_label, training=True, kt_training=True, dynamic_weight=dynamic_weight)
            
            if n_gpu > 1:
                model_loss = model_loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                model_loss = model_loss / args.gradient_accumulation_steps
            model_loss.backward() 
            
            if (step + 1) % args.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

                if scheduler is not None:
                    scheduler.step(epoch)  # Update learning rate schedule

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                # if global_step % log_step == 0 and local_rank == 0:
                if global_step % log_step == 0:
                    logger.info("Epoch: %d/%d, Step: %d/%d, Lr: %s, loss: %f,  Time/step: %f", epoch + 1,
                                args.epochs, step + 1,
                                # len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),float(model_loss), 
                                len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set([param_group['lr'] for param_group in optimizer.param_groups])))]),float(model_loss),
                                (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                    start_time = time.time()
                
            total_loss += float(model_loss)
            total_pred.append(batch_pred)
            total_true_label.append(true_label)
            total_pred_scores.append(pred_scores)
                    
        # if kt_train and epoch in [args.epochs_kt - 1, args.epochs_kt - 1-10, args.epochs_kt - 1-20, 0]:
        #     with open(os.path.join(args.output_dir, "w_misaligned"+str(epoch)+".csv"), 'w') as f:
        #         key_list = list(w_misaligned_dict.keys())
        #         writer = csv.writer(f)
        #         writer.writerow(w_misaligned_dict.keys())
        #         for i in range(len(w_misaligned_dict["t_mask"])):
        #             writer.writerow([w_misaligned_dict[x][i] for x in key_list])

        total_loss = total_loss / len(train_dataloader)
        total_pred=torch.cat(total_pred,0)
        total_true_label = torch.cat(total_true_label, 0)
        total_pred_scores = torch.cat(total_pred_scores, 0)
        
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_true_label)
        total_pred_scores = total_pred_scores.data.cpu().numpy()
        total_true_label = total_true_label.data.cpu().numpy()
        train_gap = calculate_gap(total_pred_scores, total_true_label)
        
        # if args.local_rank == 0:
        logger.info("Epoch %d/%d Finished, Train Loss: %f, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f, train_gap: %f",  \
            epoch + 1, args.epochs, total_loss, total_micro_f1, total_micro_precision, total_micro_recall,  total_acc, train_gap)
        
        # if args.local_rank == 0:
            # Validation
        logger.info("***** Running Dike valing *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch_size = %d", args.batch_size)
        # val_pred, val_label, val_pred_scores, val_loss = eval_epoch(args, model, val_dataloader, device, n_gpu, label_input, label_mask)
        
        if hasattr(model, 'module'):
            model = model.module.to(device)
        # else:
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model).to(device)
        # else:
        #     model.to(device)

        model.eval()
        with torch.no_grad():
            val_pred = []
            val_label = []
            val_pred_scores = []
            losses = []
            for _, batch in enumerate(val_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
                
                # get dynamic weight
                if args.kt_model == "Dynamic-tcp" and confidnet is not None:
                    dynamic_weight, w = get_dynamic_tcp(model, confidnet, pairs_text, pairs_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_label, device)
                elif args.kt_model == "Dynamic-ce":
                    dynamic_weight, w = get_dynamic_ce(model, confidnet, pairs_text, pairs_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_label, device)
                else:
                    dynamic_weight = None
                    
                loss, batch_pred, true_label, pred_scores = model(text, text_mask, video, video_mask, audio, audio_mask, \
                    label_input, label_mask, groundTruth_labels=groundTruth_labels, training=False, dynamic_weight=dynamic_weight)
                val_pred.append(batch_pred)
                val_label.append(true_label)
                val_pred_scores.append(pred_scores)
                losses.append(loss)
            
            val_pred=torch.cat(val_pred,0)
            val_label = torch.cat(val_label, 0)
            val_pred_scores = torch.cat(val_pred_scores, 0)
            val_loss = sum(losses) / len(losses)
            
        val_micro_f1, val_micro_precision, val_micro_recall, val_acc = get_metrics(val_pred, val_label)
        val_pred_scores = val_pred_scores.data.cpu().numpy()
        val_label = val_label.data.cpu().numpy()
        val_gap = calculate_gap(val_pred_scores, val_label)   
        val_loss = val_loss.data.cpu().numpy()
        
        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        checkpoint = Checkpoint.from_directory(args.output_dir)
        session.report({"loss": val_loss, "accuracy": val_acc, "micro_f1": val_micro_f1}, checkpoint=checkpoint)
        logger.info(tune_config)
    
        logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f, val_gap: %f", \
            val_micro_f1, val_micro_precision, val_micro_recall, val_acc, val_gap)
        output_model_file = save_model(args, model, epoch)
        
        if best_score <=  val_micro_f1:
            best_score = val_micro_f1
            best_model = model
            best_output_model_file = output_model_file
            
            torch.save(
            (model.state_dict(), optimizer.state_dict()), args.output_dir + "/checkpoint.pt")
        
        logger.info("The best model is: {}, the f1 is: {:.4f}".format(best_output_model_file, best_score))
    
    logger.info('Finished Training')

def test_best_model(best_result, args, device, n_gpu, confidnet=None):
    
    kt_loss_weight = best_result.config['kt_loss_weight']
    best_trained_model = init_model(args, device, kt_loss_weight)
    
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    # checkpoint_path = session.get_checkpoint()
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    
    train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask = prep_dataloader(args)
    label_input = label_input.to(device)
    label_mask = label_mask.to(device)

    logger.info('***** Running total testing *****')
    logger.info('  Num examples = %d', test_length)
    logger.info("  Batch_size = %d", args.batch_size)
    
    # test_pred, test_label, test_pred_scores, test_loss = eval_epoch(args, best_trained_model, test_dataloader, device, n_gpu, label_input, label_mask)

    best_trained_model.eval()
    with torch.no_grad():
        total_pred = []
        test_label = []
        test_pred_scores = []
        losses = []
        for _, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
            
            # get dynamic weight
            if args.kt_model == "Dynamic-tcp" and confidnet is not None:
                dynamic_weight, w = get_dynamic_tcp(best_trained_model, confidnet, text, text_mask, video, video_mask,audio, audio_mask, label_input, label_mask, groundTruth_labels, device)
            elif args.kt_model == "Dynamic-ce":
                dynamic_weight, w = get_dynamic_ce(best_trained_model, confidnet, text, text_mask, video, video_mask,audio, audio_mask, label_input, label_mask, groundTruth_labels, device)
            else:
                dynamic_weight = None
                    
            pred_scores, batch_pred, hidden_state, true_label, _ = best_trained_model.inference(text, text_mask, video, video_mask, audio, audio_mask, \
                label_input, label_mask, groundTruth_labels=groundTruth_labels, dynamic_weight=dynamic_weight)
            total_pred.append(batch_pred)
            test_label.append(true_label)
            test_pred_scores.append(pred_scores)
            # losses.append(loss)
        
        total_pred=torch.cat(total_pred,0)
        test_label = torch.cat(test_label, 0)
        test_pred_scores = torch.cat(test_pred_scores, 0)
        # test_loss = sum(losses) / len(losses)
    
    test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(total_pred, test_label)
    
    test_pred_scores = test_pred_scores.data.cpu().numpy()
    test_label = test_label.data.cpu().numpy()
    test_gap = calculate_gap(test_pred_scores, test_label)
    logger.info("Best trial test set result:")
    logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f, test_gap: %f", \
            test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_gap)

           
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    
    device, n_gpu = init_device(args, args.local_rank)
    
    # MER model pre-training
    if args.pretrained_model is not None:
        model = init_model(args, device)
        model.load_state_dict(torch.load(args.pretrained_model))
        best_output_model_file = args.pretrained_model
    else:
        model, best_output_model_file = train(args, device, n_gpu, n_epochs=args.epochs)
    
    ## ConfidNet Training
    if args.use_kt and args.kt_model == 'Dynamic-tcp':
        confidnet = train_confidnet(args, model, device, n_gpu, n_epochs=args.epochs_conf)
    else:
        confidnet = None
    
    ## KT Training

    if args.use_kt:
        
        # train_kt(args, device, n_gpu, pretrained_model=best_output_model_file, confidnet=confidnet, n_epochs=args.epochs_kt)
        
        num_samples = 10
        max_num_epochs = args.epochs_kt
        gpus_per_trial = 1
        
        tune_config = {
            "kt_loss_weight": tune.loguniform(1, 1e+7),
        }
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_kt, args=args, device=device, n_gpu=n_gpu, \
                    pretrained_model=best_output_model_file, confidnet=confidnet, n_epochs=args.epochs_kt),
                resources={"cpu": 1, "gpu": gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric="accuracy",
                mode="max",
                scheduler=scheduler,
                num_samples=num_samples,
            ),
            param_space=tune_config,
            run_config = RunConfig(
                local_dir='/data2/multimodal/kt_tune',
            )
        )
        result = tuner.fit()
        
        best_result = result.get_best_result("accuracy", "max")
        
        logger.info("Best trial config: {}".format(best_result.config))
        logger.info("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]))
        logger.info("Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]))
        logger.info("Best trial final validation f1_score: {}".format(
            best_result.metrics["micro_f1"]))

        test_best_model(best_result, args, device, n_gpu, confidnet=confidnet)
    
            
      
if __name__ == "__main__":
    main()


