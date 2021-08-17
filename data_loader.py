from torchvision import datasets, transforms
import torch
from torchvision import transforms
import os
from PIL import Image
import random


class ImageList(torch.utils.data.Dataset):
    '''
    Args:
        root: data root
        list_file: absolute/relative path of data list, containing labels
        transforms: image transformation function
        rand_sample: random select a sample ignoring the index
        length: required when rand_sample is True, specify the length of the dataset
        ret_anno: flag idicating whether return the label
    '''

    def __init__(self, root, list_file,transforms=None, rand_sample=False, length=None, ds="officehome"):
        self.root = root
        self.images, self.annotations = self._read_file_list(list_file,ds)
        self.rand_sample = rand_sample
        self.length = length
        self.transforms = transforms

    def __getitem__(self, idx):
        if self.rand_sample:
            idx = random.randint(0, len(self.images)-1)
        image_id = self.images[idx]

        image_path = os.path.join(self.root,image_id)   
        img = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        target = self.annotations[idx]
        target = torch.tensor(target, dtype=torch.long)
        return img, target


    def __len__(self):
        if self.rand_sample:
            return self.length
        else:
            return len(self.images)

    def _read_file_list(self, filename,ds):
        image_ids = []
        annos = []
        for line in open(filename):
            items = line.strip().split(' ')
            if ds=="imageclef":
                image_id = items[0]
                anno = int(items[1])

                domain = items[0].strip().split('/')[0]
                imgname = items[0].strip().split('/')[1]

                image_ids.append(domain+"/"+str(anno)+"/"+imgname)
                
                annos.append(anno)
            elif ds=="officehome":
                image_id = items[0]
                anno = int(items[1])
                image_ids.append(image_id)
                annos.append(anno)
            elif ds=="domainnet":
                image_id = items[0]
                anno = int(items[1])
                image_ids.append(image_id)
                annos.append(anno)
            else:
                pass


        return image_ids, annos



def load_data(args):
    folder_source_train = args.root_dir + args.source_dir + "_train.txt"
    folder_source_test = args.root_dir + args.source_dir + "_test.txt"
    folder_target_train = args.root_dir + args.target_dir + "_train.txt"
    folder_target_test = args.root_dir + args.target_dir + "_test.txt"

    train_transforms=transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    test_transforms=transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

    train_source_data = ImageList(
        args.root_dir,
        folder_source_train,
        transforms=train_transforms,
        rand_sample=args.train_rand_sample,
        length=args.num_samples,
        ds=args.dataset_name
    )

    train_target_data = ImageList(
        args.root_dir,
        folder_target_train,
        transforms=train_transforms,
        rand_sample=args.train_rand_sample,
        length=args.num_samples,
        ds=args.dataset_name
    )

    test_source_data = ImageList(
        args.root_dir,
        folder_source_test,
        transforms=test_transforms, 
        rand_sample=False,
        ds=args.dataset_name
    )

    test_target_data = ImageList(
        args.root_dir,
        folder_target_test,
        transforms=test_transforms, 
        rand_sample=False,
        ds=args.dataset_name
    )

    train_source_loader = torch.utils.data.DataLoader(
        train_source_data,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        drop_last=True
    )

    train_target_loader = torch.utils.data.DataLoader(
        train_target_data,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=None,
        num_workers=args.num_workers,
        drop_last=True
    )

    test_source_loader = torch.utils.data.DataLoader(
        test_source_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers
    )

    test_target_loader = torch.utils.data.DataLoader(
        test_target_data,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers
    )

    return train_source_loader,train_target_loader,test_source_loader,test_target_loader




