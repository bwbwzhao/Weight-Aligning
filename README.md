# Maintaining discrimination and fairness in class incremental learning

Deep neural networks (DNNs) have been applied in class incremental learning, which aims to solve common real-world problems of learning new classes continually. One drawback of standard DNNs is that they are prone to catastrophic forgetting. Knowledge distillation (KD) is a commonly used technique to alleviate this problem. In this paper, we demonstrate it can indeed help the model to output more discriminative results within old classes. However, it cannot alleviate the problem that the model tends to classify objects into new classes, causing the positive effect of KD to be hidden and limited. We observed that an important factor causing catastrophic forgetting is that the weights in the last fully connected (FC) layer are highly biased in class incremental learning. In this paper, we propose a simple and effective solution motivated by the aforementioned observations to address catastrophic forgetting. Firstly, we utilize KD to maintain the discrimination within old classes. Then, to further maintain the fairness between old classes and new classes, we propose Weight Aligning (WA) that corrects the biased weights in the FC layer after normal training process. Unlike previous work, WA does not require any extra parameters or a validation set in advance, as it utilizes the information provided by the biased weights themselves. The proposed method is evaluated on ImageNet-1000, ImageNet-100, and CIFAR-100 under various settings. Experimental results show that the proposed method can effectively alleviate catastrophic forgetting and significantly outperform state-of-the-art methods.


## Requirements
Python  
CUDA 
PyTorch  
TorchVision  
NumPy 


## Datasets
ImageNet and CIFAR-100 are used in this repository.


## Class Increment Learning

###  cifar100 
#### ours
```
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 20 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--device 2 
```
#### ce + wa
```
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 5 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--savedir 'cencs' --dis --device 2 
```
#### ce + dis
```
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 5 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--savedir 'cedis' --ncs --device 3
```

### imagenet 
#### ours
```
python main.py \
--ds 'imagenet' \
--class_num 1000 \
--lr 0.1 \
--milestones 30 60 80 90 \
--batch_size 256 \
--epochs 100 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 10 \
--data_path '/cache/imagenet/' \
--num_workers 8 \
--device 2 \
```



## Citation

If you find this repository is helpful in your research, please consider citing our paper:
```
@inproceedings{zhao2020maintaining,
  title={Maintaining discrimination and fairness in class incremental learning},
  author={Zhao, Bowen and Xiao, Xi and Gan, Guojun and Zhang, Bin and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={13208--13217},
  year={2020}
}
```
