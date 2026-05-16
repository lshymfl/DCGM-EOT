import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10    #, LSUN
#from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.LSUNbedroom import LSUNbedroom
from datasets.church import church
from torch.utils.data import Subset
import numpy as np

def get_dataset(args):
    tran_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    test_transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor()
        ])

    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10'), train=True, download=True,
                          transform=tran_transform)
        test_dataset = CIFAR10(os.path.join(args.exp, 'datasets', 'cifar10_test'), train=False, download=True,
                               transform=test_transform)

    
    elif args.dataset == "FFHQ":
        dataset = FFHQ(path=os.path.join(args.exp, 'datasets', 'FFHQ'), transform=transforms.ToTensor(),
                           resolution=args.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        #print(indices[0:50])
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
     
    elif args.dataset == "LSUNbedroom":
        dataset = LSUNbedroom(path=os.path.join(args.exp, 'datasets', 'bedroom'), transform=transforms.ToTensor(),
                           resolution=args.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        #print(indices[0:50])
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
        
    elif args.dataset == "church":
        dataset = church(path=os.path.join(args.exp, 'datasets', 'church'), transform=transforms.ToTensor(),
                           resolution=args.image_size)

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        #print(indices[0:50])
        train_indices, test_indices = indices[:int(num_items * 0.9)], indices[int(num_items * 0.9):]
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    
    return dataset, test_dataset

      

