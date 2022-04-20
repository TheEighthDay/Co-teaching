import argparse
import numpy as np
from tensorboard_logger import Logger
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import OrderedDict
import os
import math
from dataloader import load_data
from model import BaseModel
from common import parse_args,AverageMeter,mixup_data,cross_entropy_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--config', type=str, default="configs/KDDE_DDC_ResNet50.yaml",
                    help='the name of the source dir')
parser.add_argument('--run', type=int, default=1,
                    help='run')
parser.add_argument('--source', type=str, default="Art",
                    help='the name of the source')
parser.add_argument('--target', type=str, default="Clipart",
                    help='the name of the target')
parser.add_argument('--dataset', type=str, default="officehome",
                    help='dataset')
parser.add_argument('--datasetroot', type=str, default="datasets/OfficeHome",
                    help='datasetroot')
parser.add_argument('--annotation', type=str, default="concepts65.txt",
                    help='annotation')
parser.add_argument('--num_class', type=int, default=65,
                    help='num_class')
parser.add_argument('--mode', type=str, default="train",
                    help='train predict')
args = parser.parse_args()
config = parse_args(args.config,args)


class Trainer(object):
    def __init__(self,config):
        self.config = config

    def pretrain(self):
        #save_checkpoint_path
        self.save_path = os.path.join(self.config.path.collectionroot,config.dataset+"_train","Checkpoints",self.config.data.annotation,\
        self.config.source,self.config.network.model+"_"+self.config.network.backbone+"_"+self.config.source+"_"+self.config.target,\
        "run_{}".format(self.config.run))
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        #model
        self.basemodel = BaseModel(self.config.network)
        #optimizer
        self.optimizer = optim.SGD(self.basemodel.model.parameters(), lr=self.config.train.lr, momentum=self.config.train.momentum, weight_decay = self.config.train.l2_decay)
        #scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.config.train.scheduler_epoch, gamma=self.config.train.scheduler_gamma)
        #dataloader
        self.source_train_loader,  self.target_train_loader, _, _ = load_data(self.config)
        self.len_source_train_loader = len(self.source_train_loader)
        self.len_target_train_loader = len(self.target_train_loader)
        self.len_source_train_dataset = len(self.source_train_loader.dataset)
        self.len_target_train_dataset = len(self.target_train_loader.dataset)


    def train(self):
        
        for epoch in range(self.config.train.epochs):

            self.basemodel.model.train()
            iter_source, iter_target = iter(self.source_train_loader), iter(self.target_train_loader)
            train_loss = AverageMeter()
            correct = 0.0

            for batch in range(self.len_source_train_loader):
                try:
                    (_,data_source, label_source) = iter_source.next()
                    (_,data_target, label_target) = iter_target.next()
                except StopIteration:
                    iter_source, iter_target = iter(self.source_train_loader), iter(self.target_train_loader)
                    (_,data_source, label_source) = iter_source.next()
                    (_,data_target, label_target) = iter_target.next()
                data_source, label_source = data_source.cuda(), label_source.cuda()
                data_target, label_target = data_target.cuda(), label_target.cuda()

                output= self.basemodel.forward(data_source)
                target_output = self.basemodel.forward(data_target)
                self.optimizer.zero_grad()

                source_soft_label = self.basemodel.forward_sup(data_source)
                if self.config.train.kdCT is True:
                    sourcemodel_target_soft_label = self.basemodel.forward_sup(data_target)
                
                target_soft_label = self.basemodel.forward_da(data_target) 
                if self.config.train.kdCT is True:
                    targetmodel_source_soft_label = self.basemodel.forward_da(data_source)

                source_loss = cross_entropy_loss(output, source_soft_label.softmax(dim=1), temperature=self.config.train.temperature)
                target_loss = cross_entropy_loss(target_output, target_soft_label.softmax(dim=1), temperature=self.config.train.temperature)

                if (self.config.train.miCT is True) and (self.config.train.kdCT is False):
                    mix_data,mix_soft_label = mixup_data(data_source,data_target,source_soft_label.softmax(dim=1),target_soft_label.softmax(dim=1))
                    mix_output= self.basemodel.forward(mix_data)
                    mix_loss = cross_entropy_loss(mix_output,mix_soft_label)
                    loss=(0.5*source_loss+0.5*target_loss)*0.5+0.5*mix_loss
                
                elif (self.config.train.kdCT is True) and (self.config.train.miCT is False):
                    mutual_rate = np.random.beta(self.config.train.alpha,self.config.train.beta)
                    sourcemodel_target_loss = cross_entropy_loss(target_output, sourcemodel_target_soft_label.softmax(dim=1), temperature=self.config.train.temperature)
                    targetmodel_source_loss = cross_entropy_loss(output, targetmodel_source_soft_label.softmax(dim=1), temperature=self.config.train.temperature)
                    loss=(mutual_rate*source_loss+(1-mutual_rate)*targetmodel_source_loss)*0.5+(mutual_rate*target_loss+(1-mutual_rate)*sourcemodel_target_loss)*0.5
                
                elif (self.config.train.kdCT is True) and (self.config.train.miCT is True):
                    mutual_rate =np.random.beta(self.config.train.alpha,self.config.train.beta)
                    mix_data,mix_soft_label=mixup_data(data_source,data_target,source_soft_label.softmax(dim=1),target_soft_label.softmax(dim=1))
                    mix_output= self.basemodel.forward(mix_data)
                    mix_loss = cross_entropy_loss(mix_output,mix_soft_label)
                    sourcemodel_target_loss = cross_entropy_loss(target_output, sourcemodel_target_soft_label.softmax(dim=1), temperature=self.config.train.temperature)
                    targetmodel_source_loss = cross_entropy_loss(output, targetmodel_source_soft_label.softmax(dim=1), temperature=self.config.train.temperature)
                    loss=((mutual_rate*source_loss+(1-mutual_rate)*targetmodel_source_loss)*0.5+(mutual_rate*target_loss+(1-mutual_rate)*sourcemodel_target_loss)*0.5)*0.5+0.5*mix_loss

                else:
                    loss=0.5*source_loss+0.5*target_loss
                
                loss=loss.mean()
                train_loss.update(loss.item())

                loss.backward()
                self.optimizer.step()

                pred = torch.max(output, 1)[1]
                correct = correct+torch.sum(pred == label_source)

                if batch % config.train.log_interval == 0:
                    print('Train Epoch: [{}/{} ({:02d}%)], total_Loss: {:.6f}'.format(epoch + 1,self.config.train.epochs,int(100. * batch / self.len_source_train_loader ),  train_loss.avg)) 

            self.scheduler.step()

        torch.save(self.basemodel.model.state_dict(),os.path.join(self.save_path,"[{}2{}].pth".format(self.config.source[0],self.config.target[0])))

if __name__=="__main__":
    trainer = Trainer(config)
    trainer.pretrain()
    trainer.train()






