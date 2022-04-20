
# Co-Teaching for Unsupervised Domain Adaptation and Expansion

## Introduction

pass

## Data for UDA and UDE

First, we need download [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) and [DomainNet](http://ai.bu.edu/M3SDA/) into `datasets` folder.
Our data division follows the [KDDE](https://arxiv.org/abs/2104.00233). We can download division and pretrained checkpoints in this [link](xxxx).



## Inference



## Train 


## Performance


### Models


* ResNet50: Trained exclusively on the source domain.
* DDC: A classical deep domain adaptation model that minimizes domain discrepancy measured in light of first-order statistics of the deep features (Tzeng et al., Deep Domain Confusion: Maximizing for Domain Invariance, ArXiv 2014)
* SRDC:
* KDDE: 
* Co-Teaching:



### Office-Home

```
python eval_all_tasks.py --test_collection officehome_test
```


### DomainNet

```
python eval_all_tasks.py --test_collection domainnet_test
```


## Publications on UDE


Citation of the UDE task and data is the following:

```
@article{tomm-ude,      
title={Unsupervised Domain Expansion for Visual Categorization},    
author={Jie Wang and Kaibin Tian and Dayong Ding and Gang Yang and Xirong Li},     
journal = {ACM Transactions on Multimedia Computing Communications and Applications (TOMM)},   
year={2021},  
note={in press},  
}
@article{co-teaching,
author = {Tian, Kaibin and Wei, Qijie and Li, Xirong},
title = {Co-Teaching for Unsupervised Domain Adaptation and Expansion},
year = {2022},
journal = {ArXiv},
volume={abs/2204.01210}
}

```

