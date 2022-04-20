from torchvision import datasets, transforms
import torch
from torchvision import transforms
import os
from PIL import Image
import random


class ImageList(torch.utils.data.Dataset):

    def __init__(self, path, mode, domain, dataset):
        self.mode = mode
        self.domain = domain
        self.dataset = dataset
        self.collectionroot = path.collectionroot
        self.datasetroot = path.datasetroot
        self.train_transforms = transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        self.test_transforms = transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        self.image_ids, self.annotations = self._read_file_list()


    def __getitem__(self, idx):

        image_id = self.image_ids[idx]
        image_path = os.path.join(self.datasetroot,image_id)   
        img = Image.open(image_path).convert('RGB')

        if self.mode == "train":
            img = self.train_transforms(img)
        elif self.mode == "test":
            img = self.test_transforms(img) 
        else:
            raise Exception("Invalid mode")

        target = self.annotations[idx]
        target = torch.tensor(target, dtype=torch.long)
        return image_id, img, target


    def __len__(self):
        return len(self.image_ids)

    def _read_file_list(self):
        self.collectionroot,self.dataset,self.mode,self.domain
        image_ids_path = os.path.join(self.collectionroot,self.dataset+"_"+self.mode,"ImageSets",self.domain+".txt")
        f=open(image_ids_path,"r")
        image_ids=f.readlines()
        f.close()
        image_ids = [x.strip() for x in image_ids]

        annos=[]
        if self.dataset=="domainnet":
            annos_txts_rootpath = os.path.join(self.collectionroot,self.dataset+"_"+self.mode,"Annotations","Image","concepts345.txt",self.domain)
        elif self.dataset=="officehome":
            annos_txts_rootpath = os.path.join(self.collectionroot,self.dataset+"_"+self.mode,"Annotations","Image","concepts65.txt",self.domain)
        else:
            Exception("Invalid dataset")
        annos_txts_names = os.listdir(annos_txts_rootpath)
        for annos_txt in annos_txts_names:
            f=open(os.path.join(annos_txts_rootpath,annos_txt),"r")
            data=f.readlines()
            f.close()
            annos.extend([ int(x.strip().split(' ')[1]) for x in data])
        return image_ids, annos



def load_data(config):

    train_source_data = ImageList(config.path,"train",config.source,config.dataset)
    train_target_data = ImageList(config.path,"train",config.target,config.dataset)
    test_source_data = ImageList(config.path,"test",config.source,config.dataset)
    test_target_data = ImageList(config.path,"test",config.target,config.dataset)

   

    train_source_loader = torch.utils.data.DataLoader(
        train_source_data,
        batch_size=config.train.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=config.train.num_workers,
        drop_last=True
    )

    train_target_loader = torch.utils.data.DataLoader(
        train_target_data,
        batch_size=config.train.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=config.train.num_workers,
        drop_last=True
    )

    test_source_loader = torch.utils.data.DataLoader(
        test_source_data,
        batch_size=config.test.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.test.num_workers
    )

    test_target_loader = torch.utils.data.DataLoader(
        test_target_data,
        batch_size=config.test.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.test.num_workers
    )

    return train_source_loader,train_target_loader,test_source_loader,test_target_loader




