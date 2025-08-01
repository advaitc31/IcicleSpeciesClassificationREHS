import os
import time
import argparse
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
import json
from collections import defaultdict
import math
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import model
import importlib

importlib.reload(model)

from model import DistMult
from resnet import Resnet18, Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
from PIL import Image
from dataset import iWildCamOTTDataset
from pytorchtools import EarlyStopping

_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

'''
Code credit: https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/examples/train.py
'''

#################
# image to id
#################
    
def train_image_id(train_loader, model, optimizer, writer, args, epoch_id):
    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_image_id = 0.0
    criterion_ce = nn.CrossEntropyLoss()

    for labeled_batch in tqdm(train_loader['image_to_id']):
        h, r, t = labeled_batch
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, triple_type=('image', 'id'))
        # outputs = model(h)

        batch_results = {
            'y_true': t.cpu(),
            'y_pred': outputs.cpu(),
        }

        # compute objective
        loss = criterion_ce(batch_results['y_pred'], batch_results['y_true'])
        batch_results['objective'] = loss.item()
        loss.backward()

        avg_loss_image_id += loss.item()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()

        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)

        epoch_y_pred.append(y_pred)

        batch_idx += 1
        if args.debug:
            break

    avg_loss_image_id = avg_loss_image_id/len(train_loader['image_to_id'])
    print('train/avg_loss_image_id = {}'.format(avg_loss_image_id))
    writer.add_scalar('image_id_loss/train', avg_loss_image_id, epoch_id)

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)

    metrics = [
        Accuracy(prediction_fn=None),
        Recall(prediction_fn=None, average='macro'),
        F1(prediction_fn=None, average='macro'),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    
    results['epoch'] = epoch_id
    print(f'Train epoch {epoch_id}, image to id, Average acc: {results[metrics[0].agg_metric_field]*100.0:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100.0:.2f}')
    
    writer.add_scalar('acc_image_id/train', results[metrics[0].agg_metric_field]*100.0, epoch_id)
    writer.add_scalar('f1_macro_image_id/train', results[metrics[2].agg_metric_field]*100.0, epoch_id)

#################
# id to id
#################
def train_id_id(train_loader, model, optimizer, writer, args, epoch_id):
    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_id_id = 0.0
    criterion_ce = nn.CrossEntropyLoss()

    for labeled_batch in tqdm(train_loader['id_to_id']):
        h, r, t = labeled_batch
        # print(h, r, t)
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, triple_type=('id', 'id'))

        batch_results = {
            'y_true': t.cpu(),
            'y_pred': outputs.cpu(),
        }

        # compute objective
        loss = criterion_ce(batch_results['y_pred'], batch_results['y_true'])
        avg_loss_id_id += loss.item()

        # print('loss = {}'.format(loss.item()))
        batch_results['objective'] = loss.item()
        loss.backward()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()
        
        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)

        epoch_y_pred.append(y_pred)
        
        batch_idx += 1
        if args.debug:
            break

    avg_loss_id_id = avg_loss_id_id/len(train_loader['id_to_id'])
    print('avg_loss_id_id = {}'.format(avg_loss_id_id))
    writer.add_scalar('avg_loss_id_id/train', avg_loss_id_id, epoch_id)

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)

    metrics = [
        Accuracy(prediction_fn=None),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    results['epoch'] = epoch_id
    print(f'Train epoch {epoch_id}, id to id, Average acc: {results[metrics[0].agg_metric_field]*100.0:.2f}')
    writer.add_scalar('acc_id_id/train', results[metrics[0].agg_metric_field]*100.0, epoch_id)

#################
# image to location
#################
def train_image_location(train_loader, model, optimizer, writer, args, epoch_id):
    batch_idx = 0
    avg_loss_image_location = 0.0
    criterion_bce = nn.BCEWithLogitsLoss()

    for labeled_batch in tqdm(train_loader['image_to_location']):
        h, r, t = labeled_batch

        # print(h, r, t)
        # print(t)
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)
        
        outputs = model.forward_ce(h, r, triple_type=('image', 'location'))
        target = F.one_hot(t, num_classes=len(model.all_locs)).float()
        loss = criterion_bce(outputs, target)
        
        avg_loss_image_location += loss.item()

        loss.backward()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()

        batch_idx += 1
        if args.debug:
            break

    avg_loss_image_location = avg_loss_image_location/len(train_loader['image_to_location'])
    print('avg_loss_image_location = {}'.format(avg_loss_image_location))
    writer.add_scalar('avg_loss_image_location/train', avg_loss_image_location, epoch_id)

#################
# image to time
#################
def train_image_time(train_loader, model, optimizer, writer, args, epoch_id):
    batch_idx = 0
    avg_loss_image_time = 0.0
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    
    for labeled_batch in tqdm(train_loader['image_to_time']):
        h, r, t = labeled_batch

        # print(h, r, t)
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)
        
        outputs = model.forward_ce(h, r, triple_type=('image', 'time'))
        target = F.one_hot(t, num_classes=len(model.all_timestamps)).float()
        loss = criterion_bce(outputs, target)
    
        avg_loss_image_time += loss.item()

        loss.backward()

        # update model and logs based on effective batch
        optimizer.step()
        model.zero_grad()

        batch_idx += 1
        if args.debug:
            break

    avg_loss_image_time = avg_loss_image_time/len(train_loader['image_to_time'])
    print('avg_loss_image_time = {}'.format(avg_loss_image_time))
    writer.add_scalar('avg_loss_image_time/train', avg_loss_image_time, epoch_id)

def train(model, train_loader, optimizer, epoch_id, writer, args):
    model.train()
    torch.set_grad_enabled(True)
    
    if args.add_id_id:
        train_id_id(train_loader, model, optimizer, writer, args, epoch_id)
    
    if args.add_image_location:
        train_image_location(train_loader, model, optimizer, writer, args, epoch_id)

    if args.add_image_time:
        train_image_time(train_loader, model, optimizer, writer, args, epoch_id)
    
    train_image_id(train_loader, model, optimizer, writer, args, epoch_id)
   
    return

def evaluate(model, val_loader, optimizer, early_stopping, epoch_id, writer, args):
    model.eval()
    torch.set_grad_enabled(False)
    criterion = nn.CrossEntropyLoss()

    epoch_y_true = []
    epoch_y_pred = []

    batch_idx = 0
    avg_loss_image_id = 0.0
    for labeled_batch in tqdm(val_loader):
        h, r, t = labeled_batch
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, triple_type=('image', 'id'))

        batch_results = {
            'y_true': t.cpu(),
            'y_pred': outputs.cpu(),
        }

        batch_results['objective'] = criterion(batch_results['y_pred'], batch_results['y_true']).item()
        avg_loss_image_id += batch_results['objective']

        epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])
        y_pred = y_pred.argmax(-1)

        epoch_y_pred.append(y_pred)

        batch_idx += 1
        if args.debug:
            break

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)

    metrics = [
        Accuracy(prediction_fn=None),
        Recall(prediction_fn=None, average='macro'),
        F1(prediction_fn=None, average='macro'),
    ]

    results = {}

    for i in range(len(metrics)):
        results.update({
            **metrics[i].compute(epoch_y_pred, epoch_y_true),
                    })

    results['epoch'] = epoch_id

    avg_loss_image_id = avg_loss_image_id/len(val_loader)
    
    early_stopping(-1*results[metrics[0].agg_metric_field], model, optimizer)
    
    print('val/avg_loss_image_id = {}'.format(avg_loss_image_id))
    writer.add_scalar('image_id_loss/val', avg_loss_image_id, epoch_id)

    writer.add_scalar('acc_image_id/val', results[metrics[0].agg_metric_field]*100, epoch_id)
    writer.add_scalar('f1_macro_image_id/val', results[metrics[2].agg_metric_field]*100, epoch_id)

    print(f'Eval. epoch {epoch_id}, image to id, Average acc: {results[metrics[0].agg_metric_field]*100:.2f}, F1 macro: {results[metrics[2].agg_metric_field]*100:.2f}')

    return results, epoch_y_pred


def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def generate_target_list(data, entity2id):
    sub = data.loc[(data["datatype_h"] == "image") & (data["datatype_t"] == "id"), ['t']]
    sub = list(sub['t'])
    categories = []
    for item in tqdm(sub):
        if entity2id[str(int(float(item)))] not in categories:
            categories.append(entity2id[str(int(float(item)))])
    # print('categories = {}'.format(categories))
    print("No. of target categories = {}".format(len(categories)))
    return torch.tensor(categories, dtype=torch.long).unsqueeze(-1)

class AttrDict(dict):
    """ Dictionary that allows attribute access. """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def run_training(args_dict):
    args = AttrDict(args_dict)

    print('args = {}'.format(args))
    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    writer = SummaryWriter(log_dir=args.save_dir)

    datacsv = pd.read_csv(os.path.join(args.data_dir, 'dataset_subtree.csv'), low_memory=False)
    entity_id_file = os.path.join(args.data_dir, 'entity2id_subtree.json')

    if not os.path.exists(entity_id_file):
        entity2id = {}
        for i in tqdm(range(datacsv.shape[0])):
            if datacsv.iloc[i,1] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,0]))))
            if datacsv.iloc[i,-2] == "id":
                _get_id(entity2id, str(int(float(datacsv.iloc[i,-3]))))
        json.dump(entity2id, open(entity_id_file, 'w'))
    else:
        entity2id = json.load(open(entity_id_file, 'r'))

    num_ent_id = len(entity2id)
    print('len(entity2id) = {}'.format(num_ent_id))
    target_list = generate_target_list(datacsv, entity2id)

    # Datasets
    train_image_to_id_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="image", tail_type="id")
    if args.add_id_id:
        train_id_to_id_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="id", tail_type="id")
    if args.add_image_location:
        train_image_to_location_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="image", tail_type="location")
    if args.add_image_time:
        train_image_to_time_dataset = iWildCamOTTDataset(datacsv, 'train', args, entity2id, target_list, head_type="image", tail_type="time")
    val_image_to_id_dataset = iWildCamOTTDataset(datacsv, 'val', args, entity2id, target_list, head_type="image", tail_type="id")

    # DataLoaders
    train_loaders = {
        'image_to_id': DataLoader(train_image_to_id_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    }
    if args.add_id_id:
        train_loaders['id_to_id'] = DataLoader(train_id_to_id_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    if args.add_image_location:
        train_loaders['image_to_location'] = DataLoader(train_image_to_location_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    if args.add_image_time:
        train_loaders['image_to_time'] = DataLoader(train_image_to_time_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_image_to_id_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Model
    if args.add_image_time:
        model = DistMult(args, num_ent_id, target_list, args.device, train_image_to_id_dataset.all_locs, all_timestamps=train_image_to_time_dataset.all_timestamps)
    else:
        model = DistMult(args, num_ent_id, target_list, args.device, train_image_to_id_dataset.all_locs)
    model.to(args.device)

    early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True,
                                   ckpt_path=os.path.join(args.save_dir, 'model.pt'),
                                   best_ckpt_path=os.path.join(args.save_dir, 'best_model.pt'))

    optimizer = optim.Adam(
        [
            {"params": model.ent_embedding.parameters(), "lr": args.lr},
            {"params": model.rel_embedding.parameters(), "lr": args.lr},
            {"params": model.image_embedding.parameters(), "lr": 3e-5},
            {"params": model.location_embedding.parameters(), "lr": args.loc_lr},
            {"params": model.time_embedding.parameters(), "lr": args.time_lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay)

    if args.ckpt_path:
        print('ckpt loaded...')
        ckpt = torch.load(args.ckpt_path)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['dense_optimizer'])

    for epoch_id in range(args.start_epoch, args.n_epochs):
        print('\nEpoch [%d]:\n' % epoch_id)

        train(model, train_loaders, optimizer, epoch_id, writer, args)
        val_results, y_pred = evaluate(model, val_loader, optimizer, early_stopping, epoch_id, writer, args)

        if early_stopping.early_stop:
            print("Early stopping...")
            break

    writer.close()

