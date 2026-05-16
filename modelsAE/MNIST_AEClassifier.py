import torch
import torch.nn as nn
from utils.lsoftmax import LSoftmaxLinear
from torch.nn import functional as F

 
class AutoencoderClassifier(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64, class_nums=10, margin=0.35, scale=30):
        super(AutoencoderClassifier, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(dim_c, dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(dim_f),
                nn.Conv2d(dim_f, 2*dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(2*dim_f),
                nn.Conv2d(2*dim_f, 4*dim_f, 3, 2, 1),nn.PReLU(),nn.BatchNorm2d(4*dim_f),
                nn.Conv2d(4*dim_f, 8*dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(8*dim_f),
                nn.Conv2d(8*dim_f, dim_z, 2, 1, 0)
        )  

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim_z, 8*dim_f, 2, 1, 0),  
            nn.ConvTranspose2d(8*dim_f, 4*dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(4*dim_f),
            nn.ConvTranspose2d(4*dim_f, 2*dim_f, 3, 2, 1),nn.PReLU(),nn.BatchNorm2d(2*dim_f),
            nn.ConvTranspose2d(2*dim_f, dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(dim_f),
            nn.ConvTranspose2d(dim_f, dim_c, 4, 2, 1)
        )
        self.dim_f = dim_f
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.class_nums = class_nums
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(dim_z, class_nums))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        embeddings = encoded.view(encoded.size(0), -1) 
        return decoded, embeddings

    def am_softmax(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=0)
        logits = torch.matmul(embeddings, weight)
        logits_with_margin = logits - self.margin
        one_hot_labels = F.one_hot(labels, self.weight.size(1)).float()
        logits = one_hot_labels * logits_with_margin + (1 - one_hot_labels) * logits
        logits = logits * self.scale
        return logits