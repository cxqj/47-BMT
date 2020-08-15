import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from datasets.load_features import load_features_from_npy
from utilities.proposal_utils import (filter_meta_for_video_id,
                                      get_center_coords, get_segment_lengths)


class ProposalGenerationDataset(Dataset):
    # phase: train / val_1  
    def __init__(self, cfg, phase, pad_idx=1):
        '''we hardcode pad_idx to be 1'''
        self.cfg = cfg
        self.modality = cfg.modality  # audio_video
        self.phase = phase
        if self.phase == 'train':
            self.meta_path = cfg.train_meta_path  # './data/train.csv'
        elif self.phase == 'val_1':
            self.meta_path = cfg.val_1_meta_path 
        elif self.phase == 'val_2':
            self.meta_path = cfg.val_2_meta_path
        else:
            raise NotImplementedError

        self.pad_idx = pad_idx   # 1
        self.feature_names_list = []  # ['i3d_features','vggish_features']
        if 'video' in self.modality:
            self.video_feature_name = f'{cfg.video_feature_name}_features'
            self.feature_names_list.append(self.video_feature_name)
        if 'audio' in self.modality:
            self.audio_feature_name = f'{cfg.audio_feature_name}_features'
            self.feature_names_list.append(self.audio_feature_name)
        self.meta_dataset = pd.read_csv(self.meta_path, sep='\t')
        self.dataset = self.meta_dataset['video_id'].unique().tolist()  # 视频名称列表
        # self.dataset = self.meta_dataset['video_id'].sample(n=4).unique().tolist()
        # print(self.dataset)

        # the dataset filtering is done considering videos only
        # 将不满足条件的视频过滤
        print(f'Dataset size (before filtering, {phase}): {len(self.dataset)}')
        self.filtered_ids_filepath = f'./tmp/filtered_ids_from_{phase}_for{self.modality}.txt'  # 将不存在的视频ID保存到这个文件中
        self.dataset = self.filter_dataset()  # 过滤掉那些动作时长<0的视频

        if phase in ['train', 'val_1', 'val_2']:
            print(f'Dataset size (after filtering, {phase}): {len(self.dataset)}')
            self.extracted_targets_path = f'./tmp/extracted_targets_for_{phase}.pkl'
            self.dataset_targets = self.extract_targets()  # 生成target信息


    def __getitem__(self, idx):
        video_id = self.dataset[idx]  
        feature_stacks = self.get_feature_stacks(video_id)
        targets_dict = None
        # targets_dict['targets'] will be changed in collate4...(), hence, deepcopy is used
        targets_dict = copy.deepcopy(self.dataset_targets[video_id])
        """
        targets_dict:
            { 'targets': (事件数，4)--(batch_id,事件中心，事件时长，在meta_data中的索引)
              'phase': 'train'/'val_1'
              'duration': 视频时长
              'video_id': 视频名称 
            }
        """
        return feature_stacks, targets_dict

    def __len__(self):
        return len(self.dataset)

    def get_feature_stacks(self, video_id):
        """
        feature_stacks:
            {orig_feat_length: {'audio':音频特征长度，'rgb':rgb特征长度, 'flow':flow特征长度}
             'audio':(800,128)
             'rgb':(300,1024)
             'flow':(300,1024)  
            }  
        """
        feature_stacks = load_features_from_npy(
            self.cfg, self.feature_names_list, video_id,
            start=None, end=None, duration=None, pad_idx=self.pad_idx, get_full_feat=True
        )
        return feature_stacks

    def collate4proposal_generation(self, batch):
        """
        batch: [[feat_stacks,targets_dict],[feat_stacks,targets_dict],...]
        """
        """
        feature_stacks = {'rgb':(B,300,1024), 'flow':(B,300,1024), 'audio':(B,800,128)}
        """
        feature_stacks = {}
        if 'video' in self.modality:
            rgb = [features['rgb'].unsqueeze(0) for features, _ in batch]    # [(1,300,1024),(1,300,1024),...(1,300,1024)]
            flow = [features['flow'].unsqueeze(0) for features, _ in batch]  # [(1,300,1024),(1,300,1024),...(1,300,1024)]
            feature_stacks['rgb'] = torch.cat(rgb, dim=0).to(self.cfg.device)      # (B,300,1024)
            feature_stacks['flow'] = torch.cat(flow, dim=0).to(self.cfg.device)    # (B,300,1024)
        if 'audio' in self.modality:
            audio = [features['audio'].unsqueeze(0) for features, _ in batch]
            feature_stacks['audio'] = torch.cat(audio, dim=0).to(self.cfg.device)  # (B,800,128) 

        
        if self.phase in ['train', 'val_1', 'val_2']:
            for video_idx, (features, targets_dict) in enumerate(batch):
                targets_dict['targets'][:, 0] = video_idx   # 为target_dict添加batch索引

            video_ids = [targets_dict['video_id'] for _, targets_dict in batch]
            duration_in_secs = [targets_dict['duration'] for _, targets_dict in batch]
            targets = [targets_dict['targets'] for _, targets_dict in batch]
            targets = torch.cat(targets, dim=0).to(self.cfg.device)

        batch = {
            'feature_stacks': feature_stacks,  #  {'rgb':(B,300,1024), 'flow':(B,300,1024), 'audio':(B,800,128)}
            'targets': targets,      # [(事件数，4),(事件数，4),.....(事件数，4)]
            'video_ids': video_ids,  # [id1,id2,...idB]
            'duration_in_secs': duration_in_secs,  
        }

        return batch

    def filter_dataset(self):
        '''filters out a video id if any of the features is None'''
        filtered_examples = []

        if self.phase in ['train', 'val_1', 'val_2']:
            # find_faulty_segments
            cond = self.meta_dataset['end'] - self.meta_dataset['start'] <= 0
            filtered_examples += self.meta_dataset[cond]['video_id'].unique().tolist()

        if os.path.exists(self.filtered_ids_filepath):
            filtered_examples += open(self.filtered_ids_filepath, 'r').readline().split(', ')
            print(f'Loading filtered examples from: {self.filtered_ids_filepath}')
        else:
            for video_id in tqdm(self.dataset, desc=f'Filtering the dataset {self.phase}'):
                stacks = self.get_feature_stacks(video_id)
                if any(features is None for feat_name, features in stacks.items()):
                    filtered_examples.append(video_id)

            os.makedirs('./tmp/', exist_ok=True)
            with open(self.filtered_ids_filepath, 'w') as writef:
                writef.write(', '.join(filtered_examples))
                print(f'Saved filtered dataset {self.phase} @ {self.filtered_ids_filepath}')

        dataset = [vid for vid in self.dataset if vid not in filtered_examples]
        print(f'Filtered examples {self.phase}: {filtered_examples}')

        return dataset

    def extract_targets(self):
        '''
            Cols (targets):
            0th: 0s (to fill with feature id in a batch) (0000112222233 -- for 4 videos)
            1st: event centers in seconds
            2nd: event length in seconds
            3rd: idx in meta

            returns: dict[vid_id -> dict[targets, phase, duration, vid_id -> *]]
        '''

        try:
            dataset_targets = pickle.load(open(self.extracted_targets_path, 'rb'))
            print(f'Using pickled targets for {self.phase} from {self.extracted_targets_path}')
            return dataset_targets
        except FileNotFoundError:
            dataset_targets = {}

            for video_id in tqdm(self.dataset, desc='Preparing targets'):
                video_id_meta = filter_meta_for_video_id(self.meta_dataset, video_id)  # 获取属于该视频的Meta信息
                event_num = len(video_id_meta)
                start_end_numpy = video_id_meta[['start', 'end']].to_numpy()  # (N,2)
                center_coords = get_center_coords(start_end_numpy)      # (N,)  事件中心
                segment_lengths = get_segment_lengths(start_end_numpy)  # (N,)  事件时长
                duration = float(video_id_meta['duration'].unique())  # 视频时长

                meta_indices = video_id_meta['idx'].to_numpy()  # 事件在meta_data中的索引

                # 水平堆叠信息
                targets = np.column_stack([
                    np.zeros((event_num, 1)),
                    center_coords,
                    segment_lengths,
                    meta_indices,
                ])   # (N,4), 4分别用于视频名，事件中心，事件时长，事件索引

                phase = video_id_meta['phase'].unique()[0]  # train / val_1

                dataset_targets[video_id] = {
                    'targets': torch.from_numpy(targets).float(),
                    'phase': phase,
                    'duration': duration,
                    'video_id': video_id,
                }

            os.makedirs('./tmp/', exist_ok=True)
            pickle.dump(dataset_targets, open(self.extracted_targets_path, 'wb'))
            print(f'Saved pickled targets for {self.phase} @ {self.extracted_targets_path}')
            return dataset_targets
