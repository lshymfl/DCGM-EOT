import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

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
            self.activate = nn.LeakyReLU(0.2, inplace=True)
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
    def __init__(self, dim_z=256, dim_c=3, dim_f=64, class_nums=12):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, dim_f, 3, 2, 1),nn.LeakyReLU(0.2, inplace=True),)  
        self.attention1 = NonLocalBlock(dim_f)  #ChannelAttention(dim_f)
        self.rb1 = ResBlock(dim_f, 2*dim_f, 3, 2, 1, 'down') # 16,16 -->8,8
        self.attention2 = NonLocalBlock(2*dim_f)  #ChannelAttention(2*dim_f)
        self.rb2 = ResBlock(2*dim_f, 4*dim_f, 3, 2, 1, 'down') # 8,8 -->4,4 
        self.attention3 = NonLocalBlock(4*dim_f)  #ChannelAttention(4*dim_f)
        self.rb3 = ResBlock(4*dim_f, 8*dim_f, 3, 2, 1, 'down') # 8,8 -->4,4
        self.attention4 = NonLocalBlock(8*dim_f)  #ChannelAttention(8*dim_f)  
        self.conv2 = nn.Conv2d(8*dim_f, dim_z, 4, 1, 0) # 4,4 -->100 1 1 
        #self.linear = nn.Linear(int(np.prod(image_shape)), dim_z) # 100 1 1                
        #self.dropout = nn.Dropout2d(p=0.2)
        
        self.classifier = nn.Sequential(nn.Linear(dim_z, 512), nn.PReLU(), nn.Linear(512, 256), nn.PReLU(), nn.Linear(256, 128), nn.PReLU(),
                                        nn.Linear(128, class_nums), )
        
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
        #rb5 = rb5.view(inputs.shape[0],-1)
        #out = self.linear(rb5)
        endlayer = self.classifier(out.squeeze().detach())
        return out,  endlayer 
        

class Decoder(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64):
        super(Decoder, self).__init__()        
        self.convT1 = nn.ConvTranspose2d(dim_z, 8*dim_f, 4, 1, 0) #  1,1 -->4,4 
        self.deatten1 = NonLocalBlock(8*dim_f)  #ChannelAttention(8*dim_f)
        self.rb7 = ResBlock(8*dim_f, 4*dim_f, 2, 2, 0, 'up') # 4,4 -->8,8
        self.deatten1_5 = NonLocalBlock(4*dim_f)  #ChannelAttention(4*dim_f)
        self.rb8 = ResBlock(4*dim_f, 2*dim_f, 2, 2, 0, 'up') # 4,4 -->8,8
        self.deatten2 = NonLocalBlock(2*dim_f)  #ChannelAttention(2*dim_f)
        self.rb9 = ResBlock(2*dim_f, dim_f, 2, 2, 0, 'up') # 8,8 -->16,16  
        self.deatten3 = NonLocalBlock(dim_f)  #ChannelAttention(dim_f)     
        self.convT2 = nn.Sequential(nn.ConvTranspose2d(dim_f, 3, 2, 2, 0),nn.Sigmoid(),)
    
        
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
