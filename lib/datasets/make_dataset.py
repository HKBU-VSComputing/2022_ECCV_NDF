import imp
import os
import time

import numpy as np
import torch
import torch.utils.data
from IPython import embed
from lib.config.config import cfg

from . import samplers
from .collate_batch import make_collator
from .transforms import make_transforms


def _dataset_factory(is_train, current_epoch):
    if is_train:
        module = cfg.train_dataset_module
        path = cfg.train_dataset_path
        args = cfg.train_dataset
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
        args = cfg.test_dataset

    dataset = imp.load_source(module,
                              path).Dataset(**args,
                                            current_epoch=current_epoch)
    return dataset


def make_dataset(cfg,
                 dataset_name,
                 transforms,
                 is_train=True,
                 current_epoch=None):
    dataset = _dataset_factory(is_train, current_epoch)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler':
        sampler = samplers.FrameSampler(dataset)
        return sampler
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg,
                     is_train=True,
                     is_distributed=False,
                     max_iter=-1,
                     current_epoch=None):
    if is_train:
        batch_size = cfg.train.batch_size
        # shuffle = True
        shuffle = cfg.train.shuffle
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset

    transforms = make_transforms(cfg, is_train)
    dataset = make_dataset(cfg, dataset_name, transforms, is_train,
                           current_epoch)
    print('dataset test')
    dataset[0]
    print('dataset test succeed')
    sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)
    if is_train:
        num_workers = cfg.train.num_workers
    else:
        try:
            num_workers = cfg.test.num_workers
        except:
            num_workers = cfg.train.num_workers

    collator = make_collator(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn)

    return data_loader
