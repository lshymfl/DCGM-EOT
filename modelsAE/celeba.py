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
            self.activate = nn.PReLU() #nn.LeakyReLU(0.2, inplace=True)
        elif mode == 'up':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
            self.activate = nn.ReLU(inplace=True)   ###nn.ReLU(inplace=True) 
        self.BN = nn.BatchNorm2d(c_out )   ###nn.BatchNorm2d(c_out, momentum=0.1)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x)) 
        relu = self.activate(conv1)       
        conv2 = self.BN(self.conv2(relu))
        if self.resize:           
            x = self.BN(self.conv1(x))
        return self.activate(x + conv2)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y        

class Encoder(nn.Module):  
    def __init__(self, dim_z=256, dim_c=3, dim_f=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, dim_f, 3, 2, 1),nn.PReLU(),)  
        self.attention1 = ChannelAttention(dim_f)
        self.rb1 = ResBlock(dim_f, 2*dim_f, 3, 2, 1, 'down') # 16,16 -->8,8
        self.attention2 =  ChannelAttention(2*dim_f)
        self.rb2 = ResBlock(2*dim_f, 4*dim_f, 3, 2, 1, 'down') # 8,8 -->4,4 
        self.attention3 =  ChannelAttention(4*dim_f)
        self.rb3 = ResBlock(4*dim_f, 8*dim_f, 3, 2, 1, 'down') # 8,8 -->4,4
        self.attention4 = ChannelAttention(8*dim_f)  
        self.conv2 = nn.Conv2d(8*dim_f, dim_z, 4, 1, 0) # 4,4 -->100 1 1 
 
        
    def forward(self, inputs):   
        conv1 = self.conv1(inputs)
        #conv1 = self.attention1(conv1)
        rb1 = self.rb1(conv1)
        #rb1 = self.attention2(rb1) 
        rb2 = self.rb2(rb1)
        #rb2 = self.attention3(rb2) 
        rb3 = self.rb3(rb2)   
        #rb3 = self.attention4(rb3)      
        out = self.conv2(rb3)
 
        return out  #conv1,rb1,rb2,rb3,rb4,rb5,out 
        

class Decoder(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64):
        super(Decoder, self).__init__()        
        self.convT1 = nn.ConvTranspose2d(dim_z, 8*dim_f, 4, 1, 0) #  1,1 -->4,4 
        self.deatten1 =ChannelAttention(8*dim_f)
        self.rb7 = ResBlock(8*dim_f, 4*dim_f, 2, 2, 0, 'up') # 4,4 -->8,8
        self.deatten1_5 = ChannelAttention(4*dim_f)
        self.rb8 = ResBlock(4*dim_f, 2*dim_f, 2, 2, 0, 'up') # 4,4 -->8,8
        self.deatten2 = ChannelAttention(2*dim_f)
        self.rb9 = ResBlock(2*dim_f, dim_f, 2, 2, 0, 'up') # 8,8 -->16,16  
        self.deatten3 = ChannelAttention(dim_f)     
        self.convT2 = nn.Sequential(nn.ConvTranspose2d(dim_f, 3, 2, 2, 0),nn.Tanh(),)
    
        
    def forward(self, inputs):
        convT1 = self.convT1(inputs)
        #convT1 = self.deatten1(convT1)    
        rb7 = self.rb7(convT1) 
        #rb7 = self.deatten1_5(rb7)                                        
        rb8 = self.rb8(rb7)
        #rb8 = self.deatten2(rb8) 
        rb9 = self.rb9(rb8)
        #rb9 = self.deatten3(rb9)
        output = self.convT2(rb9)
        return output #linear,rb8,rb9,rb10,rb11,rb12,output


class autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self, dim_z=None, dim_c=None, dim_f=None):
        super(autoencoder, self).__init__()
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.dim_f = dim_f
        self.encoder = Encoder(self.dim_z,self.dim_c,self.dim_f)
        self.decoder = Decoder(self.dim_z,self.dim_c,self.dim_f)
  
    
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded,encoded  
