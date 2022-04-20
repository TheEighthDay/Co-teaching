import model.DIS
import model.DDC
import model.ResNet
import model.SRDC
import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import os
class BaseModel(object):
    def __init__(self, network):
        self.network = network
        self.model = self.init_model(network.model,network.model_weight)
        if network.sup_model!="None":
            self.sup_model= self.init_model(network.sup_model,network.sup_model_weight)
        if network.da_model!="None":
            self.da_model= self.init_model(network.da_model,network.da_model_weight)
        
    def init_model(self,model_name,model_weight):
        if model_name=="ResNet50":
            model = ResNet.resnet50(num_classes=self.network.num_class, pretrained=True)
        elif model_name=="DDC":
            model = DDC.DDCNet(self.network.num_class, transfer_loss='mmd', base_net='ResNet50')
        elif model_name=="SRDC":
            model = SRDC.SRDC(num_classeses=self.network.num_class,num_neurons=128)
        elif model_name=="KDDE_DDC" or model_name=="CT_DDC" or model_name=="KDDE_SRDC" or model_name=="CT_SRDC":
            model = DIS.StudentNet(num_class=self.network.num_class, base_net='ResNet50')
        else:
            Exception("invid model name")
        

        if model_weight != "None":
            model_weight = os.path.realpath(model_weight)
            pretrained_dict=torch.load(model_weight)
            try: #
                pretrained_dict=pretrained_dict["state_dict"]
            except:
                pretrained_dict=pretrained_dict
            new_state_dict = OrderedDict()
            for i,(k, v) in enumerate(pretrained_dict.items()):
                if k.startswith("module.model_resnet50."):
                    k=k.replace("module.model_resnet50.","")
                if k.startswith("module."):
                    k=k.replace("module.","")
                new_state_dict[k]=v
            model.load_state_dict(new_state_dict)
        model = nn.DataParallel(model, device_ids=self.network.gpus).cuda()
        model.eval()
        return model


    def forward(self,data):
        if self.network.model=="DDC":
            output=self.model.module.predict(data)
        elif self.network.model=="KD_DDC" or self.network.model=="CT_DDC" or self.network.model=="CT_SRDC" or self.network.model=="KD_SRDC":
            output=self.model(data)
        elif self.network.model=="SRDC":
            _, _, output = self.model(data)
        else:
            output=self.model(data)
        return output
    
    def forward_sup(self,data):
        with torch.no_grad():
            output = self.sup_model(data).detach()
        return output
    

    def forward_da(self,data):
        with torch.no_grad():
            if self.network.da_model=="DDC":
                output = self.da_model.module.predict(data).detach()
            elif self.network.da_model=="SRDC":
                _,_,output = self.da_model(data)
                output = output.detach()
            else:
                Exception("Forward da model error.")

        return output


        