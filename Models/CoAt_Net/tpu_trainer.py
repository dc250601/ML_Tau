import os
import numpy as np
import torch
import timm
import coat
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from typing import Any
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from einops import rearrange
from einops.layers.torch import Rearrange
import wandb
import PIL.Image as Image
import PIL as pil
import time
import einops
import gc


os.environ['XLA_USE_BF16']                 = '1'
os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'
batch_size = 128
num_epochs = 100
num_tpu_workers = 8
dataset_path = "../../Tau_Dataset/"
LR = 0.001
WD = 0.05

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils


wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
wandb.init(
    project = "Tau_Run1"
)
wandb.config.update({"batch_size":batch_size*num_tpu_workers,
		     "Learning Rate":LR,
		     "Weight Decay": WD

                    })

class restruct(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        """
        Args:
            img (Tensor): The stacked Image .
        Returns:
            Tensor: Restructured Image into 13 channels.
        """
    
        return einops.rearrange(torch.squeeze(img),
                                'h ( w c ) -> c h w ',
                                w = 125,
                                c=13)


def metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
 


train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            restruct(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomRotation(60),])



test_transform = transforms.Compose([
                            transforms.ToTensor(),
                            restruct()])


dataset_train = datasets.ImageFolder(os.path.join(dataset_path,"Train"),
                                    transform =train_transform,
                                    loader = pil_loader)

dataset_test = datasets.ImageFolder(os.path.join(dataset_path,"Test"),
                                    transform =test_transform,
                                    loader = pil_loader)

def train_loop_fn(data_loader, model, optimizer, device,loss_fn):
    train_loss = 0
    train_steps = 0
    model.train() # put model in training mode
    
    for image, label in tqdm(data_loader): # enumerate through the dataloader
        
        with torch.no_grad():
            image = nn.functional.pad(image, (2,1,2,1))
            
        # put tensors onto desired device, in this case the TPU core
        image = image.to(device) 
        label = label.to(device)
        
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        outputs = model(image)
        loss = loss_fn(outputs, label.float())
        train_loss += loss.item()
        train_steps +=1
                
        # backpropagate
        loss.backward()
        
        # Use PyTorch XLA optimizer stepping
        xm.optimizer_step(optimizer)
        
        
        
        
    loss_reduced = xm.mesh_reduce('loss_reduce',train_loss,lambda x:(sum(x)/len(x)))
    loss_reduced = loss_reduced/train_steps
    
    xm.master_print("Training complete----")
    xm.master_print("Training Loss",loss_reduced)
    
    if xm.is_master_ordinal(local=False):
        wandb.log({ "Train_loss_epoch": loss_reduced},commit = False)
        

def val_loop_fn(data_loader, model, device, loss_fn,scheduler = None):
    model.eval()
    val_loss = 0
    val_steps = 0
    label_list = []
    outputs_list = []
    
    for image, label in tqdm(data_loader): # enumerate through the dataloader
        
        with torch.no_grad():
            image = nn.functional.pad(image, (2,1,2,1))
            
            # put tensors onto desired device, in this case the TPU core
            image = image.to(device) 
            label = label.to(device)

            outputs = model(image)
            loss = loss_fn(outputs, label.float())
            
            label = xm.all_gather(label.detach())
            outputs = xm.all_gather(outputs.detach())
            label_list.extend(label.cpu().numpy().tolist())
            outputs_list.extend(outputs.cpu().numpy().tolist())

            
            val_loss += loss.item()
            val_steps +=1
        
    loss_reduced = xm.mesh_reduce('loss_reduce',val_loss,lambda x:(sum(x)/len(x)))
    loss_reduced = loss_reduced/val_steps
    val_auc = metric(label_list, outputs_list) 
    xm.master_print("Epoch Completed-----")
    xm.master_print("Validation loss",loss_reduced)
    xm.master_print("Validation AUC",val_auc)
    xm.master_print("--------------------")
    
    if scheduler is not None:
        scheduler.step(val_auc)
    
    if xm.is_master_ordinal(local=False):
        curr_lr = scheduler._last_lr[0]
        wandb.log({"Val_loss_epoch": loss_reduced,
                        "Val_auc_epoch": val_auc,
                        "Lr": curr_lr}, commit = False)
        

def _run(model):
    
    ### DATA PREP
    
    # data samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,
                                                                    num_replicas = xm.xrt_world_size(),
                                                                    rank         = xm.get_ordinal(),
                                                                    shuffle      = True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test,
                                                                    num_replicas = xm.xrt_world_size(),
                                                                    rank         = xm.get_ordinal(),
                                                                    shuffle      = False)
    
    # data loaders
    valid_loader = torch.utils.data.DataLoader(dataset_test, 
                                               batch_size  = batch_size, 
                                               sampler     = valid_sampler, 
                                               num_workers = 32,
                                               pin_memory  = True) 
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size  = batch_size, 
                                               sampler     = train_sampler, 
                                               num_workers = 32,
                                               pin_memory  = True)
    
    
    ### MODEL PREP
    
    # send to TPU
    device = xm.xla_device()
    model  = model.to(device)
    
    # optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = LR,
                                  weight_decay = WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max',
                                                           verbose = True,
                                                           threshold = 0.0001,
                                                           patience = 3,
                                                           factor = 0.5)
    
    
        
    # modeling loop
    for epoch in range(num_epochs):
        xm.master_print(f"Epoch {epoch} starting ...")
        
        # update train_loader shuffling
        train_loader.sampler.set_epoch(epoch)
        
        
        # training pass
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(data_loader = para_loader.per_device_loader(device), 
                      model = model,
                      optimizer = optimizer,
                      device = device,
                      loss_fn = criterion)
        

        # validation pass
        para_loader = pl.ParallelLoader(valid_loader, [device])
        
        val_loop_fn(data_loader = para_loader.per_device_loader(device), 
                    model = model,
                    device = device,
                    loss_fn = criterion,
                    scheduler=scheduler
                   )
        
        if xm.is_master_ordinal(local=False):
            wandb.log({"Epoch": epoch},commit = True)
            
        gc.collect()
        

def _mp_fn(rank, flags):
    image_size = (128,128)
    in_channels = 13
    num_blocks = [2, 2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    num_classes = 1
    model = coat.CoAtNet(image_size = image_size,
                        in_channels = in_channels,
                        num_blocks = num_blocks,
                        channels = channels,
                        num_classes = num_classes)
    
    mx  = xmp.MpModelWrapper(model)
    torch.set_default_tensor_type('torch.FloatTensor')
    _run(mx)

FLAGS = {}
num_tpu_workers = 8
o = xmp.spawn(_mp_fn, args = (FLAGS,), nprocs = num_tpu_workers, start_method = 'fork')
