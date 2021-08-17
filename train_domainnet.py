import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboard_logger import Logger
from torch.autograd import Variable
from torch.optim import lr_scheduler

import utils
from model import DANN,DAAN,DDC,DIS,RN
import data_loader
import math


from collections import OrderedDict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch KD model')

parser.add_argument('--model_name', type=str, default="KD_DDC",
                    help='model_name')
parser.add_argument('--base_model_name', type=str, default="RN",
                    help='model_name')
parser.add_argument('--da_model_name', type=str, default="DDC",
                    help='model_name')

parser.add_argument('--root_dir', type=str, default="/datastore/users/kaibin.tian/DE_experiment/OfficeHome/",
                    help='the path to load the data')
parser.add_argument('--base_model_checkpoint', type=str, default="/datastore/users/kaibin.tian/base_new/log/RN/officehome/[C].pth",
                    help='base_model_checkpoint ')
parser.add_argument('--da_model_checkpoint', type=str, default="/datastore/users/kaibin.tian/da_new/log2/DDC/officehome/[C2A].pth",
                    help='da_model_checkpoint ')
parser.add_argument('--temperature', type=int, default=1,
                    help='temperature')


parser.add_argument('--dataset_name', type=str, default="officehome",
                    help='dataset_name')
parser.add_argument('--log_filename', type=str, default="",
                    help='log_filename')
parser.add_argument('--source_dir', type=str, default="Clipart",
                    help='the name of the source dir')
parser.add_argument('--target_dir', type=str, default="Art",
                    help='the name of the test dir')
parser.add_argument('--num_class', default=65, type=int,
                    help='the number of classes')
parser.add_argument('--gpus', default=[0], type=list)
parser.add_argument('--gpu', default="cuda:0", type=str)
parser.add_argument('--train_rand_sample', default=False, type=bool)
parser.add_argument('--num_samples', type=int, default=2000,
                    help='num_samples')


parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='num_workers')
parser.add_argument('--epochs', type=int, default=101, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--mixup', type=bool, default=False,
                    help='start mixup')
parser.add_argument('--mutual', type=bool, default=True,
                    help='start mutual teacher')
parser.add_argument('--savedir', type=str, default="mutual_epochbeta_try1",
                    help='save dir')
parser.add_argument('--scheduler_size', type=int, default=30,
                    help='scheduler_size')
                    

args = parser.parse_args()
args.log_filename='/datastore/users/kaibin.tian/DE_experiment/kd_new/'+args.savedir+'/'+args.model_name+'/'+args.dataset_name+'/'
if not os.path.exists(args.log_filename):
    os.makedirs(args.log_filename)
args.source_dir=args.source_dir.strip()
args.target_dir=args.target_dir.strip()

def fun_alpha(x):
    #warmup
    if(x<=0.2):
        return 45*x+1
    else:
        return 10


def fun_beta(x):
    #warmup
    if(x<=0.2):
        return 10-45*x
    else:
        return 1



def train(epoch,logger,source_train_loader, target_train_loader, model,base_model,da_model, optimizer):
    #-------每个epoch前的预先工作-------------------

    if args.mutual is True:
        #mutual_rate=float(epoch/args.epochs)
        #mutual_rate=float(epoch/args.epochs)**(1/2) #0.5
        alpha=fun_alpha(float(epoch/args.epochs))
        beta=fun_beta(float(epoch/args.epochs))
    

    

    len_source_train_loader = len(source_train_loader)
    len_target_train_loader = len(target_train_loader)

    len_source_train_dataset = len(source_train_loader.dataset)
    len_target_train_dataset = len(target_train_loader.dataset)

    # print(len_target_train_dataset ,len_source_train_dataset)
    # print(len_source_train_loader ,len_target_train_loader)

    iter_source, iter_target = iter(source_train_loader), iter(target_train_loader)

    train_loss = utils.AverageMeter()
    correct = 0.0



    for batch in range(len_source_train_loader):
        #-------加载batch数据---------

        try:
            (data_source, label_source) = iter_source.next()
            (data_target, label_target) = iter_target.next()
        except StopIteration:
            iter_source, iter_target = iter(source_train_loader), iter(target_train_loader)
            (data_source, label_source) = iter_source.next()
            (data_target, label_target) = iter_target.next()
        

        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = data_target.cuda(), label_target.cuda()
        # value_source = Variable(torch.zeros((data_source.size()[0])).long().cuda())
        # value_target = Variable(torch.ones((data_target.size()[0])).long().cuda())

        #------模型处理数据（主要修改的地方）---------
        output= model(data_source)
        target_output= model(data_target)
        optimizer.zero_grad()

        if args.base_model_name=="RN":
            with torch.no_grad():
                source_soft_label = base_model(data_source).detach()
                if args.mutual is True:
                    sourcemodel_target_soft_label = base_model(data_target).detach()
        
        source_loss = utils.cross_entropy_loss(output, source_soft_label.softmax(dim=1), temperature=args.temperature)

        if args.da_model_name=="DANN":
            with torch.no_grad():
                target_soft_label = da_model(data_target,data_target,1)[0].detach()
                if args.mutual is True:
                    targetmodel_source_soft_label = da_model(data_source,data_source,1).detach()

        elif args.da_model_name=="DAAN":
            with torch.no_grad():
                target_soft_label = da_model(data_target,data_target,label_target,1)[0].detach()
                if args.mutual is True:
                    targetmodel_source_soft_label = da_model(data_source,data_source,data_source,1)[0].detach()

        elif args.da_model_name=="DDC":
            with torch.no_grad():
                target_soft_label = da_model.module.predict(data_target).detach()
                if args.mutual is True:
                    targetmodel_source_soft_label = da_model.module.predict(data_source).detach()
        else:
            pass

        target_loss = utils.cross_entropy_loss(target_output, target_soft_label.softmax(dim=1), temperature=args.temperature)

        if (args.mixup is True) and (args.mutual is False):
            mix_data,mix_soft_label=utils.mixup_data(data_source,data_target,source_soft_label.softmax(dim=1),target_soft_label.softmax(dim=1))
            mix_output= model(mix_data)
            mix_loss = utils.cross_entropy_loss(mix_output,mix_soft_label)
            loss=(0.5*source_loss+0.5*target_loss)*0.5+0.5*mix_loss
        elif (args.mutual is True) and (args.mixup is False):

            mutual_rate =np.random.beta(alpha,beta)
            sourcemodel_target_loss = utils.cross_entropy_loss(target_output, sourcemodel_target_soft_label.softmax(dim=1), temperature=args.temperature)
            targetmodel_source_loss = utils.cross_entropy_loss(output, targetmodel_source_soft_label.softmax(dim=1), temperature=args.temperature)
            loss=(mutual_rate*source_loss+(1-mutual_rate)*targetmodel_source_loss)*0.5+(mutual_rate*target_loss+(1-mutual_rate)*sourcemodel_target_loss)*0.5
        
        elif (args.mutual is True) and (args.mixup is True):
            mutual_rate = np.random.beta(alpha,beta)
            mix_data,mix_soft_label=utils.mixup_data(data_source,data_target,source_soft_label.softmax(dim=1),target_soft_label.softmax(dim=1))
            mix_output= model(mix_data)
            mix_loss = utils.cross_entropy_loss(mix_output,mix_soft_label)
            sourcemodel_target_loss = utils.cross_entropy_loss(target_output, sourcemodel_target_soft_label.softmax(dim=1), temperature=args.temperature)
            targetmodel_source_loss = utils.cross_entropy_loss(output, targetmodel_source_soft_label.softmax(dim=1), temperature=args.temperature)
            loss=((mutual_rate*source_loss+(1-mutual_rate)*targetmodel_source_loss)*0.5+(mutual_rate*target_loss+(1-mutual_rate)*sourcemodel_target_loss)*0.5)*mutual_rate+(1-mutual_rate)*mix_loss

        else:
            loss=0.5*source_loss+0.5*target_loss

        loss=loss.mean()

        train_loss.update(loss.item())

        loss.backward()
        optimizer.step()

        #--------实时打印损失情况-----------------
        pred = torch.max(output, 1)[1]
        correct = correct+torch.sum(pred == label_source)

        if batch % args.log_interval == 0:
            print('Train Epoch: [{}/{} ({:02d}%)], total_Loss: {:.6f}'.format(
                epoch + 1,
                args.epochs,
                int(100. * batch / len_source_train_loader ),  train_loss.avg)) 

    #--------每个epoch结束工作-------------
    logger.log_value("Accuracy",float(correct)/float(len_source_train_dataset),epoch)
    print('source_train_data: max correct: {}/{}, accuracy{: .2f}%\n'.format(
        correct,len_source_train_dataset, 100.0 * float(correct) / float(len_source_train_dataset)))



def test(logger, model, test_loader,epoch):
    #------测试前的预先工作-------------
    correct = 0.0
    len_test_dataset = len(test_loader.dataset)

    #----------测试每个batch（主要修改的地方）-----------
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.cuda(), label.cuda()

            output=model(data)

            pred = torch.max(output, 1)[1]
            correct = correct+torch.sum(pred == label)

            
    logger.log_value("Accuracy",float(correct)/float(len_test_dataset),epoch)
    print('test_data: max correct: {}/{}, accuracy{: .2f}%\n'.format(
         correct,len_test_dataset, 100.0 * float(correct) / float(len_test_dataset)))
    return 100.0 * float(correct) / float(len_test_dataset)

if __name__ == '__main__':
    print('Src: %s, Tar: %s distill ing .....' % (args.source_dir, args.target_dir))
    #------------------logger配置------------------
    log_source_train_path=args.log_filename+'source_train['+args.source_dir[0]+'2'+args.target_dir[0]+']'
    log_source_test_path=args.log_filename+'source_test['+args.source_dir[0]+'2'+args.target_dir[0]+']'
    #log_target_train_path=args.log_filename+'target_train['+args.source_dir[0]+'2'+args.target_dir[0]+']'
    log_target_test_path=args.log_filename+'target_test['+args.source_dir[0]+'2'+args.target_dir[0]+']'

    logger_source_train=Logger(log_source_train_path,flush_secs=20)
    logger_source_test=Logger(log_source_test_path,flush_secs=20)
    #logger_target_train=Logger(log_target_train_path,flush_secs=20)
    logger_target_test=Logger(log_target_test_path,flush_secs=20)


    #--------------------cuda配置-----------------
    # DEVICE = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    torch.cuda.manual_seed(args.seed)
    # kwargs = {'num_workers': args.num_workers, 'pin_memory': True}


    #--------------------模型加载以及训练参数设置-----------------
    if args.da_model_name=='DANN':
        da_model = eval(args.da_model_name).DANNNet(num_classes=args.num_class, base_net='ResNet50')
        pretrained_dict=torch.load(args.da_model_checkpoint)
        new_state_dict = OrderedDict()
        for i,(k, v) in enumerate(pretrained_dict.items()):
            #ks=k.split(".")
            #k=".".join(ks[1:])
            #if "model_resnet50." in k:
            #    k=k[15:]
            if k.startswith("module."):
                k=k[7:]
            new_state_dict[k]=v
        da_model.load_state_dict(new_state_dict)
        da_model=nn.DataParallel(da_model, device_ids=args.gpus).cuda()

    elif args.da_model_name=='DAAN':
        da_model = eval(args.da_model_name).DAANNet(num_classes=args.num_class, base_net='ResNet50')
        pretrained_dict=torch.load(args.da_model_checkpoint)
        new_state_dict = OrderedDict()
        for i,(k, v) in enumerate(pretrained_dict.items()):
            #ks=k.split(".")
            #k=".".join(ks[1:])
            #if "model_resnet50." in k:
            #    k=k[15:]
            if k.startswith("module."):
                k=k[7:]
            new_state_dict[k]=v
        da_model.load_state_dict(new_state_dict)
        da_model=nn.DataParallel(da_model, device_ids=args.gpus).cuda()

    elif args.da_model_name=='DDC':
        da_model = eval(args.da_model_name).DDCNet(args.num_class, transfer_loss='mmd', base_net='ResNet50')
        pretrained_dict=torch.load(args.da_model_checkpoint)
        new_state_dict = OrderedDict()
        for i,(k, v) in enumerate(pretrained_dict.items()):
            #ks=k.split(".")
            #k=".".join(ks[1:])
            #if "model_resnet50." in k:
            #    k=k[15:]
            if k.startswith("module."):
                k=k[7:]
            new_state_dict[k]=v
        da_model.load_state_dict(new_state_dict)
        da_model=nn.DataParallel(da_model, device_ids=args.gpus).cuda()

    else:
        pass


    args.lr=0.005
    model = DIS.StudentNet(num_class=args.num_class, base_net='ResNet50')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay = args.l2_decay)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_size, gamma=0.1)


    if args.base_model_name=="RN":
        base_model = getattr(RN,"resnet50")(num_classes=args.num_class, pretrained=True)
        pretrained_dict=torch.load(args.base_model_checkpoint)
        new_state_dict = OrderedDict()
        if args.dataset_name=="domainnet":
            pretrained_dict=pretrained_dict["state_dict"]
        for i,(k, v) in enumerate(pretrained_dict.items()): #["state_dict"]

            
            if k.startswith("module.model_resnet50."):
                k=k.replace("module.model_resnet50.","")


            new_state_dict[k]=v
        base_model.load_state_dict(new_state_dict)
        base_model=nn.DataParallel(base_model, device_ids=args.gpus).cuda()
    else:
        pass

    


    #--------------------数据加载-------------------

    source_train_loader,target_train_loader,source_test_loader,target_test_loader = data_loader.load_data(args)


    #--------------------训练测试------------------

    for epoch in range(args.epochs):
        model.train()
        train(epoch,logger_source_train,source_train_loader,target_train_loader,model,base_model,da_model,optimizer)
        if epoch%args.log_interval==0:
            model.eval()
            correct1=test(logger_source_test, model, source_test_loader,epoch)
            correct2=test(logger_target_test, model, target_test_loader,epoch)
        scheduler.step()


    #---------------------模型保存----------------------

    torch.save(model.state_dict(), args.log_filename+'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')

    print('modelname: %s,Src: %s, Tar: %s' % (args.model_name,args.source_dir, args.target_dir))

    
 
    

