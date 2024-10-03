import torchvision
from torchvision import transforms

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adamax, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import CyclicLR, CosineAnnealingLR, ReduceLROnPlateau, _LRScheduler

from sklearn.metrics import roc_curve

import numpy as np

import pandas as pd

from dataset import *
from models import *

import os
import math



# Ensure the directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to create data loaders accodring to the selected image size and batch_size
def make_data_loaders(train_csv,val_csv,test_csv,image_dir,batch_size,image_size):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize
    ])

    val_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    normalize
    ])

    train_dataset = ChestXRay(df_dir=train_csv, image_dir=image_dir, transform=train_transforms)
    val_dataset = ChestXRay(df_dir=val_csv, image_dir=image_dir, transform=val_transforms)
    test_dataset = ChestXRay(df_dir=test_csv, image_dir=image_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, {'train':len(train_dataset),'val':len(val_dataset),'test':len(test_dataset)}, train_dataset.class_count

# Code extracted from MIT licensed code Copyright (c) 2022 Naoki Katsura 
# Implementation from "github pytorch-cosine-annealing-with-warmup"
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    

# Code extracted from MIT licensed code Copyright (c) 2021 Alinstein Jose
# Implementation of "Weakly supervised Classification and Localization of Chest X-ray images"
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold']) # [0]??


# Idea extacted from the [https://arxiv.org/pdf/1901.05555] paper "Class-Balanced Loss Based on Effective Number of Samples"
# We can normalize inverse frequencies but this method is studied to do better in other works
def effective_weights(class_counts, beta=0.999):
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)  # Normalize the weights -> or other(?)
    return torch.tensor(weights, dtype=torch.float32)  # Convert to tensor


# Code extracted from MIT licensed code Copyright (c) 2020 Alibaba-MIIL
# Original idea in the [https://arxiv.org/pdf/2009.14119] paper "Asymmetric Loss For Multi-Label Classification"
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, average=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.average = average

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # Really small change introduced by thesis author (me) to try average loss impact
        # As discussed in "https://github.com/Alibaba-MIIL/ASL/issues/22" it's not suposed
        # To have an impact but I want to be able to use mean for ease of comparison.
        if self.average:
            return -loss.mean()
        else:
            return -loss.sum()
    
# Function to select optimizer
def get_optimizer(params, optimizer='Adam', lr=1e-4, momentum=0.9, weight_decay=0.0):
    """
    Loads and returns the selected optimizer
    """
    if optimizer == 'SGD':
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'SGD_Nesterov':
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif optimizer == 'Adamax':
        return Adamax(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdamW':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'RMSprop':
        return RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(optimizer))
    
# Function to select loss function
def get_loss(loss_type='asl1', counts=None, device='cpu', beta=0.99):

    if loss_type == 'bce':
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'bce_w':
        if counts is not None:
            class_weights = effective_weights(counts, beta=beta).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            print("You didn't provide weights so we proceed with normal BCE")
            loss_fn = nn.BCEWithLogitsLoss()
    elif loss_type == 'asymmetric':
        loss_fn = AsymmetricLoss()
    elif loss_type == 'asymmetric_avg':
        loss_fn = AsymmetricLoss(average=True)
    elif loss_type == 'asl1':
        loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0)
    elif loss_type == 'asl2':
        loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.025)
    elif loss_type == 'asl3':
        loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    elif loss_type == 'focal':
        loss_fn = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)
    
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'.")
    
    return loss_fn

# Function to select model
def get_model(model_name='dense121', pretrained=True):

    if model_name == 'res18':
        model = ResNet18(num_classes=14, pretrained=pretrained)
    elif model_name == 'res50':
        model = ResNet50(num_classes=14, pretrained=pretrained)
    elif model_name == 'dense121':
        model = DenseNet121(num_classes=14, pretrained=pretrained)
    elif model_name == 'efficientb0':
        model = EfficientNetB0(num_classes=14, pretrained=pretrained)
    elif model_name == 'efficientb3':
        model = EfficientNetB3(num_classes=14, pretrained=pretrained)
    
    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")
    
    return model


# Function to select scheduler
def get_scheduler(optimizer, name='cyclic'):

    if name == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    elif name == 'plateau1':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
    elif name == 'cyclic':
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=500, mode='triangular')
    elif name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    elif name == 'warmupcosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                          first_cycle_steps=12,
                                          cycle_mult=1,
                                          max_lr=0.01,
                                          min_lr=0.0001,
                                          warmup_steps=3,
                                          gamma=0.8)
    else:
        raise ValueError(f"Unknown scheduler type '{name}'.")
    
    return scheduler