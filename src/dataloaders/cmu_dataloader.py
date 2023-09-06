# %%
import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
import numpy as np
from collections import defaultdict
import json
import random
import time
import pickle


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

"""
CMU-MOSEI info
Train 16326 samples
Val 1871 samples
Test 4659 samples
CMU-MOSEI feature shapes
visual: (60, 35)
audio: (60, 74)
text: GLOVE->(60, 300)
label: (6) -> [happy, sad, anger, surprise, disgust, fear] 
    averaged from 3 annotators
unaligned:
text: (50, 300)
visual: (500, 35)
audio: (500, 74)    
"""

emotion_dict = {4:0, 5:1, 6:2, 7:3, 8:4, 9:5}
class AlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, zero_label_process=False):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type, zero_label_process)

    def _get_data(self, data_type, zero_label_process=False):
        data = torch.load(self.data_path)

        data = data[data_type]
        if zero_label_process:
            print("get zero label processed dataset for confidnet...")
            data = self._get_zero_label_process(data)
            
        visual = data['src-visual'].astype(np.float32)
        audio = data['src-audio'].astype(np.float32)
        text = data['src-text'].astype(np.float32)
        labels = data['tgt']
        return visual, audio, text, labels
    
    def _get_zero_label_process(self, data):
        labels = data['tgt']
        zero_label_index = []
        for i in range(len(labels)):
            label_list = labels[i]
            label = np.zeros(6, dtype=np.float32)
            filter_label = label_list[1:-1]
            for emo in filter_label:
                label[emotion_dict[emo]] =  1
            if np.sum(label) == 0:
                zero_label_index.append(i)
            
        data['src-visual'] = np.delete(data['src-visual'], zero_label_index, axis=0).astype(np.float32)
        data['src-audio'] = np.delete(data['src-audio'], zero_label_index, axis=0).astype(np.float32)
        data['src-text'] = np.delete(data['src-text'], zero_label_index, axis=0).astype(np.float32)
        data['tgt'] = np.delete(data['tgt'], zero_label_index, axis=0)
        return data
    
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] =  1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label


class UnAlignedMoseiDataset(Dataset):
    def __init__(self, data_path, data_type, zero_label_process=False):
        self.data_path = data_path
        self.data_type = data_type
        self.visual, self.audio, \
            self.text, self.labels = self._get_data(self.data_type)

    def _get_data(self, data_type, zero_label_process=False):
        label_data = torch.load(self.data_path)
        label_data = label_data[data_type]
        with open('/data2/multimodal/processed_data/mosei_senti_data_noalign.pkl', 'rb') as f:
            data = pickle.load(f)
        data = data[data_type]
        
        if zero_label_process:
            print("get zero label processed dataset for confidnet...")
            data, label_data = self._get_zero_label_process(data, label_data)
            
        visual = data['vision'].astype(np.float32)
        audio = data['audio'].astype(np.float32)
        text = data['text'].astype(np.float32)
        audio = np.array(audio)
        labels = label_data['tgt']      
        return visual, audio, text, labels
    
    def _get_zero_label_process(self, data, label_data):
        labels = label_data['tgt']
        zero_label_index = []
        for i in range(len(labels)):
            label_list = labels[i]
            label = np.zeros(6, dtype=np.float32)
            filter_label = label_list[1:-1]
            for emo in filter_label:
                label[emotion_dict[emo]] =  1
            if np.sum(label) == 0:
                zero_label_index.append(i)
            
        data['visual'] = np.delete(data['visual'], zero_label_index, axis=0).astype(np.float32)
        data['audio'] = np.delete(data['audio'], zero_label_index, axis=0).astype(np.float32)
        data['text'] = np.delete(data['text'], zero_label_index, axis=0).astype(np.float32)
        label_data['tgt'] = np.delete(label_data['tgt'], zero_label_index, axis=0)
        return data, label_data
    
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.visual[index]
        visual_mask = [1] * 50
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * 50

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_labels(self, index):
        label_list = self.labels[index]
        label = np.zeros(6, dtype=np.float32)
        filter_label = label_list[1:-1]
        for emo in filter_label:
            label[emotion_dict[emo]] =  1

        return label

    def _get_label_input(self):
        labels_embedding = np.arange(6)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)

        return labels_embedding, labels_mask
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        text, text_mask = self._get_text(index)
        visual, visual_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        label = self._get_labels(index)

        return text, text_mask, visual, visual_mask, \
            audio, audio_mask, label
            
            
class IEMOCAP_Datasets(Dataset):
    def __init__(self, dataset_path='/home/gkook/multimodal/Multimodal-Transformer/data', data='iemocap', split_type='train', if_align=True):
        super(IEMOCAP_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, data+'_data.pkl' if if_align else data+'_data_noalign.pkl' )
        dataset = pickle.load(open(dataset_path, 'rb'))

        # These are torch tensors
        #self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
        #self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
        #self.audio = dataset[split_type]['audio'].astype(np.float32)
        #self.audio[self.audio == -np.inf] = 0
        #self.audio = torch.tensor(self.audio).cpu().detach()
        #self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()
        
        self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32))
        self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32))
        self.audio = dataset[split_type]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio)
        self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32))
        
        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None
       
        self.data = data
        
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    
    def _get_text(self, index):
        text = self.text[index]
        text_mask = [1] * text.shape[0]

        text_mask = np.array(text_mask)

        return text, text_mask
    
    def _get_visual(self, index):
        visual = self.vision[index]
        visual_mask = [1] * visual.shape[0]
        visual_mask = np.array(visual_mask)

        return visual, visual_mask
    
    def _get_audio(self, index):
        audio = self.audio[index]
        audio[audio == -np.inf] = 0
        audio_mask = [1] * audio.shape[0]

        audio_mask =  np.array(audio_mask)

        return audio, audio_mask
    
    def _get_label_input(self):
        labels_embedding = np.arange(4)
        labels_mask = [1] * labels_embedding.shape[0]
        labels_mask = np.array(labels_mask)
        labels_embedding = torch.from_numpy(labels_embedding)
        labels_mask = torch.from_numpy(labels_mask)
        
        return labels_embedding, labels_mask
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        META = (0,0,0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        if self.data == 'mosi':
            META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
        if self.data == 'iemocap':
            Y = torch.argmax(Y, dim=-1).type(torch.FloatTensor)
        #return X, Y, META
        text, text_mask = self._get_text(index)
        video, video_mask = self._get_visual(index)
        audio, audio_mask = self._get_audio(index)
        return text, text_mask, video, video_mask, audio,audio_mask, Y


def get_data(args, dataset, split='train', zero_label_process=False):
    
    if args.data == "mosei":
        Dataset = AlignedMoseiDataset if args.aligned else UnAlignedMoseiDataset
    else:
        Dataset = IEMOCAP_Datasets
        
    if dataset == "iemocap":
        data_path = os.path.join(args.data_path, 'iemocap') + f'_{split}_{args.aligned}.dt'
        
        if not os.path.exists(data_path):
            print(f"  - Creating new {split} data")
            data = Dataset(args.data_path, dataset, split, args.aligned)
            torch.save(data, data_path)
        else:
            print(f"  - Found cached {split} data")
            data = torch.load(data_path)
        
    elif dataset == "mosei":
        data = Dataset(args.data_path, split, zero_label_process)
    
    return data

def prep_dataloader(args, zero_label_process=False):
    
    train_data = get_data(args, args.data, 'train', zero_label_process)
    valid_data = get_data(args, args.data, 'valid', zero_label_process)
    test_data = get_data(args, args.data, 'test', zero_label_process)
    
    label_input, label_mask = train_data._get_label_input()

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device='cuda').manual_seed(args.seed)
    )
    val_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device='cuda').manual_seed(args.seed)
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device='cuda').manual_seed(args.seed)
    )
    train_length = len(train_data)
    val_length = len(valid_data)
    test_length = len(test_data)
    
    return train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask