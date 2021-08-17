import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import utils
from model import DANN,DAAN,DDC,DIS,RN,CDAN,SRDC
import data_loader
import math
import pandas as pd
import argparse
import os
from tqdm import tqdm
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='PyTorch KD model')

parser.add_argument('--model_name', type=str, default="TMT_SRDC",  #TMT_DDC
                    help='model_name')
parser.add_argument('--dataset_name', type=str, default="officehome",  #domainnet
                    help='dataset_name')
parser.add_argument('--root_dir', type=str, default="/datastore/users/kaibin.tian/DE_experiment/OfficeHome/", #/datastore/users/kaibin.tian/DE_experiment/domainnet/
                    help='the path to load the data')
parser.add_argument('--log_filename', type=str, default="",
                    help='log_filename')
parser.add_argument('--source_dir', type=str, default="Art", #clipart Product Art
                    help='the name of the source dir')
parser.add_argument('--target_dir', type=str, default="Clipart", #painting Real_World Clipart
                    help='the name of the test dir')
parser.add_argument('--num_class', default=65, type=int, #345
                    help='the number of classes')
parser.add_argument('--gpus', default=[0], type=list)
parser.add_argument('--train_rand_sample', default=False, type=bool)
parser.add_argument('--num_samples', type=int, default=10000,
                    help='num_samples')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_workers', type=int, default=2,
                    help='num_workers')

parser.add_argument('--savedir', type=str, default="logitsandlabelsandfeatures",
                    help='save dir')
args = parser.parse_args()

def test(model,test_loader,out_put):
    #------测试前的预先工作-------------
    correct = 0.0
    len_test_dataset = len(test_loader.dataset)
    print(len_test_dataset)
    preds=[]
    labels=[]
    scores=[]
    features=[]

    #----------测试每个batch（主要修改的地方）-----------
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data, label = data.cuda(), label.cuda()
            
            if args.model_name=="DDC":
                output=model.module.predict(data)
                feature=model.module.features.cpu()
            elif args.model_name=="KD_DDC" or args.model_name=="TMT_DDC" or args.model_name=="TMT_SRDC" or args.model_name=="KD_SRDC":
                output=model(data)
                feature=model.module.features.cpu()
            elif args.model_name=="SRDC":
                feature, _, output = model(data)
                feature=feature.cpu()
            else:
                output=model(data)
                feature=model.features.cpu()

            pred = torch.max(output, 1)[1]
            correct = correct+torch.sum(pred == label)
            scores.append(output.cpu().data.numpy())
            features.append(feature.data.numpy())
            preds.append(pred.cpu().data.numpy().flatten().astype(np.int32))
            labels.append(label.cpu().data.numpy().astype(np.int32))

    save_preds=np.concatenate(preds,0)
    save_labels=np.concatenate(labels,0)
    save_scores=np.concatenate(scores,0)
    save_features=np.concatenate(features,0)

    pd.DataFrame({"preds":save_preds,"labels":save_labels}).to_csv(os.path.join(out_put,"result.csv"),index=False)
    np.save(os.path.join(out_put,"logit.npy"),save_scores)
    np.save(os.path.join(out_put,"label.npy"),save_labels)
    np.save(os.path.join(out_put,"feature.npy"),save_features)


    print('test_data: max correct: {}/{}, accuracy{: .2f}%\n'.format(
         correct,len_test_dataset, 100.0 * float(correct) / float(len_test_dataset)))
    return 100.0 * float(correct) / float(len_test_dataset)


if args.model_name=="RN":
    model = getattr(RN,"resnet50")(num_classes=args.num_class, pretrained=True)
    if args.dataset_name=="domainnet":
        model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/base_new_wj/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'].pth')
    else:
        model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/base_new/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    if args.dataset_name=="domainnet":
        pretrained_dict=pretrained_dict["state_dict"]
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module.model_resnet50."):
            k=k.replace("module.model_resnet50.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    base_model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()


if  args.model_name=="KD_DDC":
    model = DIS.StudentNet(num_class=args.num_class, base_net='ResNet50')
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/kd_new/log/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module."):
            k=k.replace("module.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

if  args.model_name=="TMT_DDC":
    model = DIS.StudentNet(num_class=args.num_class, base_net='ResNet50')
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/kd_new/mutual_epochbeta_mixup/',"KD_DDC","domainnet",'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module."):
            k=k.replace("module.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

if  args.model_name=="KD_CDAN": #wangjie
    model = getattr(RN,"resnet50")(num_classes=args.num_class, pretrained=True)
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/kd_new/log/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict["state_dict"].items()):
        if k.startswith("module.model_resnet50."):
            k=k.replace("module.model_resnet50.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    base_model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()


if args.model_name=="DDC":
    model = eval(args.model_name).DDCNet(args.num_class, transfer_loss='mmd', base_net='ResNet50')
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/da_new/log/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module."):
            k=k[7:]
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

if args.model_name=="CDAN":
    model = eval(args.model_name).CDAN(num_classes=args.num_class,fc_hidden_dim=256)
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/da_new/log/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict["state_dict"].items()):
        if k.startswith("module."):
            k=k[7:]
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=model.net_G 
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

elif args.model_name=="SRDC":
    model = eval(args.model_name).SRDC(num_classeses=args.num_class,num_neurons=128)
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/da_new/log/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    pretrained_dict=pretrained_dict["state_dict"]
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module."):
            k=k[7:]
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

if  args.model_name=="KD_SRDC":
    model = DIS.StudentNet(num_class=args.num_class, base_net='ResNet50')
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/kd_new/log/',args.model_name,args.dataset_name,'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module."):
            k=k.replace("module.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()

if  args.model_name=="TMT_SRDC":
    model = DIS.StudentNet(num_class=args.num_class, base_net='ResNet50')
    model_path=os.path.join('/datastore/users/kaibin.tian/DE_experiment/kd_new/mutual_epochbeta_mixup/',"KD_SRDC","officehome",'['+args.source_dir[0]+'2'+args.target_dir[0]+'].pth')
    pretrained_dict=torch.load(model_path)
    new_state_dict = OrderedDict()
    for i,(k, v) in enumerate(pretrained_dict.items()):
        if k.startswith("module."):
            k=k.replace("module.","")
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)
    model=nn.DataParallel(model, device_ids=args.gpus).cuda()
    model.eval()



source_train_loader,target_train_loader,source_test_loader,target_test_loader = data_loader.load_data(args)

output_source_test=os.path.join('/datastore/users/kaibin.tian/DE_experiment/analyse_transferability/',args.savedir,args.model_name,args.dataset_name,'source_test['+args.source_dir[0]+'2'+args.target_dir[0]+']')
output_target_test=os.path.join('/datastore/users/kaibin.tian/DE_experiment/analyse_transferability/',args.savedir,args.model_name,args.dataset_name,'target_test['+args.source_dir[0]+'2'+args.target_dir[0]+']')

if not os.path.exists(output_source_test):
    os.makedirs(output_source_test)

if not os.path.exists(output_target_test):
    os.makedirs(output_target_test)


test(model,source_test_loader,output_source_test)
test(model,target_test_loader,output_target_test)
