import argparse
import yaml
from common import parse_args,save_score
from dataloader import load_data
from model import BaseModel
from tqdm import tqdm
import torch
import numpy as np
import os
import torch.nn.functional as F
 

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--config', type=str, default="configs/KDDE_DDC_ResNet50.yaml",
                    help='the path of config')
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
parser.add_argument('--mode', type=str, default="predict",
                    help='train predict')
args = parser.parse_args()

config = parse_args(args.config,args)


#2 检查各种的预测结果是否准确

if config.network.model_weight=="None":
    Exception("No weights.")


class Predictor(object):
    def __init__(self,config):
        self.config=config

    def predict(self):
        # load test data
        _, _, source_test_loader, target_test_loader = load_data(self.config)

        # model
        self.basemodel=BaseModel(self.config.network)

        # save scores path
        source_save_path = os.path.join(self.config.path.collectionroot,self.config.dataset+"_test","Predictions",self.config.data.annotation,\
        self.config.source,self.config.network.model+"_"+self.config.network.backbone+"_"+self.config.source+"_"+self.config.target,"run_{}".format(self.config.run))
        target_save_path = os.path.join(self.config.path.collectionroot,self.config.dataset+"_test","Predictions",self.config.data.annotation,\
        self.config.target,self.config.network.model+"_"+self.config.network.backbone+"_"+self.config.source+"_"+self.config.target,"run_{}".format(self.config.run))

        if not os.path.exists(source_save_path):
            os.makedirs(source_save_path)
        if not os.path.exists(target_save_path):
            os.makedirs(target_save_path)
        
        source_image_ids,source_indexs,source_scores = self.get_result(source_test_loader)
        target_image_ids,target_indexs,target_scores = self.get_result(target_test_loader)

        save_score(os.path.join(source_save_path,"id.concept.score.txt"),source_image_ids,source_indexs,source_scores,self.config.dataset)
        save_score(os.path.join(target_save_path,"id.concept.score.txt"),target_image_ids,target_indexs,target_scores,self.config.dataset)
    
    def get_result(self,test_loader):
        idxs=[]
        preds=[]
        scores=[]
        features=[]
        with torch.no_grad():
            for idx, data, label in tqdm(test_loader):
                data, label = data.cuda(), label.cuda()
                output = self.basemodel.forward(data)
                output = F.softmax(output,1)
                pred = torch.max(output, 1)[1]
                scores.append(output.cpu().data.numpy())
                preds.append(pred.cpu().data.numpy().flatten().astype(np.int32))
                idxs.append(idx)
        save_preds=np.concatenate(preds,0)
        save_idxs=np.concatenate(idxs,0)
        save_scores=np.concatenate(scores,0)
        save_scores=np.array([x[y] for x,y in zip(save_scores,save_preds)])
        return save_idxs,save_preds,save_scores


if __name__=="__main__":
    predictor=Predictor(config)
    predictor.predict()
    
    







