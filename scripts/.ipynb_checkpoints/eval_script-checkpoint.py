import os
import time
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

sys.path.append('../')


from model import DistMult

from resnet import Resnet18, Resnet50

from tqdm import tqdm
from utils import collate_list, detach_and_clone, move_to
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1
from PIL import Image
from dataset import iWildCamOTTDataset

class AttrDict(dict):
    """ Dictionary that allows attribute access like an object. """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def evaluate(model, val_loader, target_list, args):
    model.eval()
    torch.set_grad_enabled(False)

    epoch_y_true = []
    epoch_y_pred = []
    epoch_y_probs=[]
    

    for labeled_batch in tqdm(val_loader):
        h, r, t = labeled_batch
        h = move_to(h, args.device)
        r = move_to(r, args.device)
        t = move_to(t, args.device)

        outputs = model.forward_ce(h, r, triple_type=('image', 'id'))


        y_true = detach_and_clone(t.cpu())
        y_pred = detach_and_clone(outputs.cpu()).argmax(-1)
        probs=F.softmax(outputs,dim=1).detach().cpu()


        epoch_y_true.append(y_true)
        epoch_y_pred.append(y_pred)
        epoch_y_probs.append(probs)

        if args.debug:
            break

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_y_probs=collate_list(epoch_y_probs)

    metrics = [
        Accuracy(prediction_fn=None),
        Recall(prediction_fn=None, average='macro'),
        F1(prediction_fn=None, average='macro'),
    ]

    results = {}
    for metric in metrics:
        results.update(metric.compute(epoch_y_pred, epoch_y_true))

    acc = results[metrics[0].agg_metric_field] * 100
    f1 = results[metrics[2].agg_metric_field] * 100

    print(f"Eval., split: {args.split}, image to id, "
          f"Average acc: {acc:.2f}, F1 macro: {f1:.2f}")

    return {       # ✅ Add this return
        "accuracy": acc,
        "f1_macro": f1,
        "y_true": epoch_y_true,
        "y_pred": epoch_y_pred,
        "y_probs": epoch_y_probs.numpy(),

    }
    

def _get_id(dict, key):
    if key not in dict:
        dict[key] = len(dict)
    return dict[key]

def generate_target_list(data, entity2id):
    sub = data.loc[(data["datatype_h"] == "image") & (data["datatype_t"] == "id"), ['t']]
    sub = list(sub['t'])
    categories = []
    for item in tqdm(sub):
        eid = entity2id[str(int(float(item)))]
        if eid not in categories:
            categories.append(eid)
    print("No. of target categories = {}".format(len(categories)))
    return torch.tensor(categories, dtype=torch.long).unsqueeze(-1)

def run_eval(args_dict):
    args = AttrDict(args_dict)

    print('args = {}'.format(args))
    args.device = torch.device('cuda') if not args.no_cuda and torch.cuda.is_available() else torch.device('cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    val_image_to_id_dataset = iWildCamOTTDataset(
        datacsv, args.split, args, entity2id, target_list,
        head_type="image", tail_type="id"
    )

    print('len(val_image_to_id_dataset) = {}'.format(len(val_image_to_id_dataset)))

    val_loader = DataLoader(
        val_image_to_id_dataset,
        shuffle=False,
        sampler=None,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True
    )

    model = DistMult(args, num_ent_id, target_list, args.device, val_image_to_id_dataset.all_locs)
    model.to(args.device)

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt['model'], strict=False)
        print('ckpt loaded...')

    eval_results = evaluate(model, val_loader, target_list, args)


    return eval_results

