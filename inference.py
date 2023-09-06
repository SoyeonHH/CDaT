from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import argparse
from src.models.models_tailor import TAILOR
from src.models.models_tfn import TFN
from src.models.models_early import Early
from src.models.models_misa import MISA
from src.models.optimization import BertAdam
from src.models.confidnet import ConfidenceRegressionNetwork, Confidnet3Layers, Confidnet4Layers
from torch.utils.data import DataLoader
import torch.utils.data as data
from util import parallel_apply, get_logger, get_tcp_target
from train import get_dynamic_tcp, get_dynamic_ce
from src.dataloaders.cmu_dataloader import get_data
from src.utils.eval import get_metrics
from collections import defaultdict
import csv
import pickle

mosei_data_dir = '/data2/multimodal/train_valid_test.pt'
iemocap_data_dir = '/data2/multimodal/IEMOCAP'

global logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader_test(args):
    test_dataset = get_data(args, args.data, 'test')
    label_input, label_mask = test_dataset._get_label_input()
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        # num_workers=4,
        # pin_memory=False,
        shuffle=args.shuffle,
        # drop_last=True,
        generator=torch.Generator(device=device).manual_seed(args.seed)
    )
    test_length = len(test_dataset)
    return  label_input, label_mask, test_dataloader, test_length

def load_model(args, n_gpu, device, model_file=None, kt_loss_weight=0.0):
    logger.info("**** loading model_file=%s *****", model_file)

    if os.path.exists(model_file):
        if args.use_kt:
            model_state_dict, optimizer_state = torch.load(model_file, map_location='cpu')
        else:
            model_state_dict = torch.load(model_file, map_location='cpu')

        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        if args.model == "tailor":
            model = TAILOR.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model, args.decoder_model, \
                task_config=args, kt_loss_weight=kt_loss_weight, device=device)
            hidden_size = args.hidden_size * args.num_classes
        elif args.model == "tfn":
            model =  TFN(args, (128, 32, 32), 64, (0.3, 0.3, 0.3, 0.3), 128, device, kt_loss_weight=kt_loss_weight)
            hidden_size = 128
        elif args.model == "early":
            model = Early(args, device, kt_loss_weight=kt_loss_weight)
            hidden_size = args.text_dim + args.video_dim + args.audio_dim
        elif args.model == "misa":
            model = MISA(args, device, kt_loss_weight=kt_loss_weight)
            hidden_size = args.hidden_size * 6
        elif args.model == "amp":
            model = MMERModel(args, device, d_model=args.hidden_size, kt_loss_weight=kt_loss_weight)
            hidden_size = args.hidden_size * args.num_classes
        elif args.model == "ours":
            model = Ours(args, device, kt_loss_weight=kt_loss_weight)
            hidden_size = args.hidden_size * args.num_classes
        
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        logger.info('***** loading model successful! *****')
        
        if args.confidnet_file != "":
            # confidnet = ConfidenceRegressionNetwork(args, hidden_size).to(device)
            # confidnet = Confidnet3Layers(args, hidden_size).to(device)
            confidnet = Confidnet4Layers(args, hidden_size).to(device)
            confidnet.load_state_dict(torch.load(args.confidnet_file, map_location='cpu'))
        else:
            confidnet = None
    else:
        model = None
        confidnet = None
    return model, confidnet

def model_test(model, confidnet, test_dataloader, device, label_input, label_mask, kt_loss_weight=0.0):
    model.eval()
    results = defaultdict(list)
    w_misalinged = defaultdict(list)
    label_input = label_input.to(device)
    label_mask = label_mask.to(device)
    with torch.no_grad():
        total_pred = []
        total_true_label = []
        idx = 0
        for _, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, ground_truth_labels = batch
            true_label = ground_truth_labels.view(-1, args.num_classes)
            
            if args.kt_model == "Dynamic-tcp" and confidnet is not None:
                dynamic_weight, w = get_dynamic_tcp(model, confidnet, text, text_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_truth_labels, device)
            elif args.kt_model == "Dynamic-ce":
                dynamic_weight, w = get_dynamic_ce(model, confidnet, text, text_mask, video, video_mask,audio, audio_mask, label_input, label_mask, ground_truth_labels, device)
            else:
                dynamic_weight, w = None, None
            
            # if w is not None:
            #     w_misalinged["t_mask"].extend(w[0].detach().cpu().numpy())
            #     w_misalinged["v_mask"].extend(w[1].detach().cpu().numpy())
            #     w_misalinged["a_mask"].extend(w[2].detach().cpu().numpy())
                
            # batch_logit, batch_pred, hidden_rep, true_label = model.inference(text, text_mask, video, video_mask, audio, audio_mask, \
            #     label_input, label_mask, groundTruth_labels=ground_truth_labels)
            batch_logit, batch_pred, h, _, feats = model.inference(text, text_mask, \
                video, video_mask, audio, audio_mask, label_input, label_mask)
            
            total_pred.append(batch_pred)
            total_true_label.append(true_label)
            
            pred_list, tcp_list = get_making_results(model, text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, ground_truth_labels)
            
            batch_size = ground_truth_labels.size(0)
            index = [i for i in range(idx, idx+batch_size)]
            idx += batch_size

            results["index"].extend(index)
            results["label"].extend(ground_truth_labels.detach().cpu().numpy())
            results["prediction"].extend(batch_pred.detach().cpu().numpy())
            results["predicted_scores"].extend(batch_logit.detach().cpu().numpy())
            # results["att_weight"].extend(att_weight.detach().cpu().numpy())
            if w is not None:
                results["w_t_mask"].extend(w[0])
                results["w_v_mask"].extend(w[1])
                results["w_a_mask"].extend(w[2])
            
            results["pred_AV"].extend(pred_list[0].cpu().numpy())
            results["pred_TA"].extend(pred_list[1].cpu().numpy())
            results["pred_TV"].extend(pred_list[2].cpu().numpy())
            results["pred_T"].extend(pred_list[3].cpu().numpy())
            results["pred_V"].extend(pred_list[4].cpu().numpy())
            results["pred_A"].extend(pred_list[5].cpu().numpy())
            
            results["tcp_TVA"].extend(get_tcp_target(ground_truth_labels, batch_logit).detach().cpu().numpy())
            results["tcp_AV"].extend(tcp_list[0].detach().cpu().numpy())
            results["tcp_TA"].extend(tcp_list[1].detach().cpu().numpy())
            results["tcp_TV"].extend(tcp_list[2].detach().cpu().numpy())
            results["tcp_T"].extend(tcp_list[3].detach().cpu().numpy())
            results["tcp_V"].extend(tcp_list[4].detach().cpu().numpy())
            results["tcp_A"].extend(tcp_list[5].detach().cpu().numpy())
            
            results["feat_t"].extend(feats[0].detach().cpu().numpy())
            results["feat_v"].extend(feats[1].detach().cpu().numpy())
            results["feat_a"].extend(feats[2].detach().cpu().numpy())
        
        total_pred=torch.cat(total_pred,0)
        total_true_label = torch.cat(total_true_label, 0)
    
    return  total_pred, total_true_label, results

def get_making_results(model, text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, ground_trunth_labels):
    logit_t_removed, pred_t_removed, _, _, _ = model.inference(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
        masked_modality=["text"], groundTruth_labels=ground_trunth_labels)
    logit_v_removed, pred_v_removed, _, _, _ = model.inference(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
        masked_modality=["video"], groundTruth_labels=ground_trunth_labels)
    logit_a_removed, pred_a_removed, _, _, _ = model.inference(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
        masked_modality=["audio"], groundTruth_labels=ground_trunth_labels)
    logit_t_only, pred_t_only, _, _, _ = model.inference(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
        masked_modality=["video", "audio"], groundTruth_labels=ground_trunth_labels)
    logit_v_only, pred_v_only, _, _, _ = model.inference(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
        masked_modality=["text", "audio"], groundTruth_labels=ground_trunth_labels)
    logit_a_only, pred_a_only, _, _, _ = model.inference(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, \
        masked_modality=["text", "video"], groundTruth_labels=ground_trunth_labels)
    
    pred_list = [pred_t_removed, pred_v_removed, pred_a_removed, pred_t_only, pred_v_only, pred_a_only]
    tcp_list = [
        get_tcp_target(ground_trunth_labels, logit_t_removed),
        get_tcp_target(ground_trunth_labels, logit_v_removed),
        get_tcp_target(ground_trunth_labels, logit_a_removed),
        get_tcp_target(ground_trunth_labels, logit_t_only),
        get_tcp_target(ground_trunth_labels, logit_v_only),
        get_tcp_target(ground_trunth_labels, logit_a_only)
    ]
    return pred_list, tcp_list


parser = argparse.ArgumentParser(description="model interfence")
parser.add_argument("--model", default="tailor", type=str, help="one of tailor, tfn, early, misa, amp")
# parser.add_argument("--data_path", default="/data2/multimodal/train_valid_test.pt", type=str, help='cmu_mosei data_path')
parser.add_argument("--model_file", default="", type=str, help="model store path")
parser.add_argument("--confidnet_file", default="", type=str, help="confidnet model store path")
parser.add_argument("--output_dir", default="/home/soyeon/workspace/Dike/checkpoint", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
# parser.add_argument("--do_test", action='store_true', help="whether to run test")
parser.add_argument("--aligned", action='store_true', help="whether train align of unalign dataset")
parser.add_argument("--data", type=str, default="mosei", help="dataset")
parser.add_argument("--shuffle", type=bool, default=True, help="whether to shuffle the data")
parser.add_argument('--num_thread_reader', type=int, default=0, help='') 
parser.add_argument('--max_words', type=int, default=60, help='')
parser.add_argument('--max_frames', type=int, default=60, help='')
parser.add_argument('--max_sequence', type=int, default=60, help='')
parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
parser.add_argument('--audio_model', default="audio-base", type=str, required=False, help='AUdio module')
parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
parser.add_argument("--bert_model", default="bert-base", type=str, required=False,
                        help="Bert pre-trained model")
parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
parser.add_argument("--num_labels", type=int, default=6, required=False)
parser.add_argument('--video_dim', type=int, default=35, required=False,help='video feature dimension')
parser.add_argument('--audio_dim', type=int, default=74, required=False, help='')
parser.add_argument('--text_dim', type=int, default=300, help='text_feature_dimension') 
parser.add_argument('--bert_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
parser.add_argument('--visual_num_hidden_layers', type=int, default=4, help="Layer NO. of visual.")
parser.add_argument('--audio_num_hidden_layers', type=int, default=4, help="Layer NO. of audio")
parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")
parser.add_argument("--num_classes", default=6, type=int, required=False)
parser.add_argument("--hidden_size",type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--seed', type=int, default=42, help='random seed') 
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
parser.add_argument('--use_kt', action='store_true', help='whether to use knowledge transfer')
parser.add_argument('--kt_model', type=str, default='Dynamic-tcp', help='knowledge transfer model')
args = parser.parse_args()
n_gpu = 1

kt_model_name = {'Static': 'const', 'Dynamic-ce': 'ce', 'Dynamic-tcp': 'confidnet'}
if args.use_kt:
    args.output_dir = os.path.join(args.output_dir, f'{args.data}_{args.model}_{kt_model_name[args.kt_model]}')
else:
    args.output_dir = os.path.join(args.output_dir, f'{args.data}_{args.model}')
        
if args.data == "mosei":
    args.data_path = mosei_data_dir
elif args.data == "iemocap":
    args.data_path = iemocap_data_dir
    args.num_classes = 4
    
args.n_gpu = 1

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
logger = get_logger(os.path.join(args.output_dir, "log.txt"))

start=time.time()
if args.local_rank ==0:
    model, confidnet = load_model(args, n_gpu, device, model_file=args.model_file)
    logger.info("***** dataloader loading *****")
    label_input, label_mask, test_dataloader, test_length = dataloader_test(args)
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", test_length)
    logger.info("  Batch size = %d", 64)
    logger.info("  Num steps = %d", len(test_dataloader)) 
    total_pred, total_true_label, results = model_test(model, confidnet, test_dataloader, device, label_input, label_mask)
    
    with open(os.path.join(args.output_dir, "model_test_results.csv"), 'w') as f:
        key_list = list(results.keys())
        writer = csv.writer(f)
        writer.writerow(results.keys())
        for i in range(len(results["index"])):
            writer.writerow([results[x][i] for x in key_list])
    
    with open(os.path.join(args.output_dir, "model_test_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    
    test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(total_pred, total_true_label)
    logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f", \
                    test_micro_f1, test_micro_precision, test_micro_recall, test_acc)
    logger.info("inference time: {}".format(time.time() - start))