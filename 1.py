import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import pickle as pkl

class Registry:
    def __init__(self):
        self._modules = {}

    def register_module(self, module):
        def decorator(cls):
            self._modules[module] = cls
            return cls
        return decorator

DATASETS = Registry()

class Process:
    def __init__(self, processes, cfg):
        self.processes = processes
        self.cfg = cfg

    def __call__(self, sample):
        for process in self.processes:
            sample = process(sample, self.cfg)
        return sample

class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)
        self.load_annotations()

    def load_annotations(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        history = np.array(data_info['history'], dtype=np.float32)
        future = np.array(data_info['future'], dtype=np.float32)
        planning_features = np.array(data_info['planning_features'], dtype=np.float32)
        target = np.array(data_info['target'], dtype=np.float32)
        in_drivable_area = np.array(data_info['in_drivable_area'], dtype=np.bool_)
        distance_error = np.array(data_info['distance_error'], dtype=np.float32)

        sample = {
            'history': history,
            'future': future,
            'planning_features': planning_features,
            'target': target,
            'in_drivable_area': in_drivable_area,
            'distance_error': distance_error
        }

        if self.training:
            sample = self.processes(sample)

        return sample

@DATASETS.register_module(module='ArgoverseDataset')
class ArgoverseDataset(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.list_path = osp.join(data_root, f'{split}_list.txt')
        self.split = split
        super().__init__(data_root, split, processes=processes, cfg=cfg)

    def load_annotations(self):
        self.logger.info('Loading Argoverse annotations...')
        cache_path = f'cache/argoverse_{self.split}.pkl'
        os.makedirs('cache', exist_ok=True)
        if osp.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
            return

        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                self.data_infos.append(self.load_annotation(line.strip()))

        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        info = {}
        parts = line.split()
        info['history'] = np.load(osp.join(self.data_root, parts[0]))
        info['future'] = np.load(osp.join(self.data_root, parts[1]))
        info['planning_features'] = np.load(osp.join(self.data_root, parts[2]))
        info['target'] = np.load(osp.join(self.data_root, parts[3]))
        info['in_drivable_area'] = np.load(osp.join(self.data_root, parts[4]))
        info['distance_error'] = np.load(osp.join(self.data_root, parts[5]))
        return info

