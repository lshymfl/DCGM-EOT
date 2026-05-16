import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


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
        

class Encoder(nn.Module):  
    def __init__(self, dim_z=256, dim_c=3, dim_f=64, class_nums=4):
        super(Encoder, self).__init__()
        self.init_conv = nn.Sequential(
            # [-1, 3, 256, 256] -> [-1, 128, 128, 128]
            nn.Conv2d(3, dim_f, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
                                       ) 
        self.BN = nn.BatchNorm2d(dim_f) 
        self.rb0 = ResBlock(dim_f, 2*dim_f, 3, 2, 1, 'down') # 128,128 -->64,64
        self.rb1 = ResBlock(2*dim_f, 4*dim_f, 3, 2, 1, 'down') # 64,64 -->32,32
        self.rb2 = ResBlock(4*dim_f, 8*dim_f, 3, 2, 1, 'down') # 32,32 -->16,16
        self.rb3 = ResBlock(8*dim_f, 16*dim_f, 3, 2, 1, 'down') # 16,16 -->8,8
        self.rb4 = ResBlock(16*dim_f, 32*dim_f, 3, 2, 1, 'down') # 8,8 -->4,4        
        self.rb5 = nn.Conv2d(32*dim_f, dim_z, 4, 1, 0) # 100 1 1 
        
        self.classifier = nn.Sequential(nn.Linear(dim_z, 1024), nn.PReLU(), nn.Linear(1024, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), 
                                        nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, class_nums), )         
       
    def forward(self, inputs):   
        init_conv = self.init_conv(inputs)
        rb0 = self.rb0(init_conv)
        rb1 = self.rb1(rb0)        
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        out = self.rb5(rb4)
        endlayer = self.classifier(out.squeeze().detach())
        return out,  endlayer
        

class Decoder(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64 ):
        super(Decoder, self).__init__()        
        self.conTrans = nn.ConvTranspose2d(dim_z, 32*dim_f, 4, 1, 0) #  1,1 -->4,4         
        self.rb6 = ResBlock(32*dim_f, 16*dim_f, 2, 2, 0, 'up') # 4,4 -->8,8
        self.rb7 = ResBlock(16*dim_f, 8*dim_f, 2, 2, 0, 'up') # 8,8 -->16,16       
        self.rb8 = ResBlock(8*dim_f, 4*dim_f, 2, 2, 0, 'up') # 16,16 -->32,32  
        self.rb9 = ResBlock(4*dim_f, 2*dim_f, 2, 2, 0, 'up') # 32,32-->64,64
        self.rb10 = ResBlock(2*dim_f, dim_f, 2, 2, 0, 'up') # 64,64-->128,128
        self.out_conv = nn.Sequential(nn.ConvTranspose2d(dim_f, 3, 2, 2, 0),nn.Sigmoid(),)
        
        
        
    def forward(self, inputs):
        iniTrans = self.conTrans(inputs)
        rb6 = self.rb6(iniTrans)                                               
        rb7 = self.rb7(rb6)
        rb8 = self.rb8(rb7)
        rb9 = self.rb9(rb8)
        rb10 = self.rb10(rb9)
        output = self.out_conv(rb10)
        return output


class autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self, dim_z=None, dim_c=None, dim_f=None, class_nums=None):
        super(autoencoder, self).__init__()
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.dim_f = dim_f
        self.class_nums = class_nums
        self.encoder = Encoder(self.dim_z,self.dim_c,self.dim_f, self.class_nums)
        self.decoder = Decoder(self.dim_z,self.dim_c,self.dim_f)
  
    
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded, endlayer = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, endlayer, decoded 
