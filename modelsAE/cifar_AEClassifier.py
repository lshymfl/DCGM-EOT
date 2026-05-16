import torch
import torch.nn as nn
from utils.lsoftmax import LSoftmaxLinear
from torch.nn import functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels

        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        #  theta, phi, g
        theta = self.theta(x).view(batch_size, channels // 2, height * width)
        phi = self.phi(x).view(batch_size, channels // 2, height * width)
        g = self.g(x).view(batch_size, channels // 2, height * width)

        attention = torch.matmul(theta.transpose(1, 2), phi)
        attention = nn.functional.softmax(attention, dim=-1)

       
        fusion = torch.matmul(g, attention.transpose(1, 2))
        fusion = fusion.view(batch_size, channels // 2, height, width)

        #  
        fusion = self.out_conv(fusion)

        #  
        output = x + fusion

        return output 

 

class AutoencoderClassifier(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64, class_nums=10, margin=0.35, scale=30):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, dim_f, 3, 2, 1),nn.LeakyReLU(0.2, inplace=True), 
                nn.Conv2d(dim_f, 2*dim_f, 3, 2, 1),nn.LeakyReLU(0.2, inplace=True), 
                nn.Conv2d(2*dim_f, 4*dim_f, 3, 2, 1),nn.LeakyReLU(0.2, inplace=True), 
                nn.Conv2d(4*dim_f, dim_z, 4, 1, 0),
        )  

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim_z, 4*dim_f, 4, 1, 0),NonLocalBlock(4*dim_f),  
            nn.ConvTranspose2d(4*dim_f, 2*dim_f, 2, 2, 0),nn.ReLU(inplace=True),NonLocalBlock(2*dim_f),
            nn.ConvTranspose2d(2*dim_f, dim_f, 2, 2, 0),nn.ReLU(inplace=True),NonLocalBlock(dim_f),
            nn.ConvTranspose2d(dim_f, 3, 2, 2, 0),nn.Sigmoid(),
        )
        self.dim_f = dim_f
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.class_nums = class_nums
        self.scale = scale
        self.margin = margin
        #self.linear = nn.Sequential(nn.Linear(dim_z, 128), nn.ReLU(), ) 
        #self.weight = nn.Parameter(torch.FloatTensor(128, class_nums)) 
        self.weight = nn.Parameter(torch.FloatTensor(dim_z, class_nums))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        embeddings = encoded.view(encoded.size(0), -1) 
        return decoded, embeddings

    def am_softmax(self, embeddings, labels):
        weight = F.normalize(self.weight, p=2, dim=0)
        #embed = self.linear(embeddings)
        #embeds = F.normalize(embed, p=2, dim=1)
        #logits = torch.matmul(embeds, weight)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = torch.matmul(embeddings, weight)
        logits_with_margin = logits - self.margin
        one_hot_labels = F.one_hot(labels, self.weight.size(1)).float()
        logits = one_hot_labels * logits_with_margin + (1 - one_hot_labels) * logits
        logits = logits * self.scale
        return logits