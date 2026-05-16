import torch
import torch.nn as nn
from utils.lsoftmax import LSoftmaxLinear
from torch.nn import functional as F

class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='down'):  ###  k,kernel//s, strides//p,padding
        assert mode in ['down', 'up'], "Mode must be either 'down' or 'up'."
        super(ResBlock, self).__init__()
        if mode == 'down':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'up':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.activate = nn.LeakyReLU(0.2, inplace=True)   ###nn.ReLU(inplace=True) 
        self.BN = nn.BatchNorm2d(c_out)   ###nn.BatchNorm2d(c_out, momentum=0.1)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x)) 
        relu = self.activate(conv1)       
        conv2 = self.BN(self.conv2(relu))
        if self.resize:           
            x = self.BN(self.conv1(x))
        return self.activate(x + conv2)

 

class AutoencoderClassifier(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64, class_nums=10, margin=0.35, scale=30):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, dim_f, 3, 2, 1),nn.LeakyReLU(0.2, inplace=True), 
                ResBlock(dim_f, 2*dim_f, 3, 2, 1, 'down'),
                ResBlock(2*dim_f, 4*dim_f, 3, 2, 1, 'down'), 
                ResBlock(4*dim_f, 8*dim_f, 3, 2, 1, 'down'),
                ResBlock(8*dim_f, 16*dim_f, 3, 2, 1, 'down'),
                ResBlock(16*dim_f, 32*dim_f, 3, 2, 1, 'down'),
                nn.Conv2d(32*dim_f, dim_z, 4, 1, 0)
        )  

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim_z, 32*dim_f, 4, 1, 0),  
            ResBlock(32*dim_f, 16*dim_f, 2, 2, 0, 'up'),
            ResBlock(16*dim_f, 8*dim_f, 2, 2, 0, 'up'),
            ResBlock(8*dim_f, 4*dim_f, 2, 2, 0, 'up'),
            ResBlock(4*dim_f, 2*dim_f, 2, 2, 0, 'up'),
            ResBlock(2*dim_f, dim_f, 2, 2, 0, 'up'),
            nn.ConvTranspose2d(dim_f, 3, 2, 2, 0),nn.Sigmoid(),
        )
        self.dim_f = dim_f
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.class_nums = class_nums
        self.scale = scale
        self.margin = margin
        self.linear = nn.Sequential(nn.Linear(dim_z, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 128), nn.PReLU(), )  
        self.weight = nn.Parameter(torch.FloatTensor(128, class_nums))
        #self.weight = nn.Parameter(torch.FloatTensor(dim_z, class_nums))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        embeddings = encoded.view(encoded.size(0), -1) 
        return decoded, embeddings

    def am_softmax(self, embeddings, labels):
        weight = F.normalize(self.weight, p=2, dim=0)
        embed = self.linear(embeddings)
        embeds = F.normalize(embed, p=2, dim=1)
        logits = torch.matmul(embeds, weight)
        #embeddings = F.normalize(embeddings, p=2, dim=1)
        #logits = torch.matmul(embeddings, weight)
        logits_with_margin = logits - self.margin
        one_hot_labels = F.one_hot(labels, self.weight.size(1)).float()
        logits = one_hot_labels * logits_with_margin + (1 - one_hot_labels) * logits
        logits = logits * self.scale
        return logits