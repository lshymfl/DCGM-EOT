import torch
import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Function
#from torch.autograd import Variable
#import pdb

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

class Encoder(nn.Module):  
    def __init__(self, dim_z=256, dim_c=3, dim_f=64):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(dim_f),)  
        self.conv2 = nn.Sequential(nn.Conv2d(dim_f, 2*dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(2*dim_f),)
        self.conv3 = nn.Sequential(nn.Conv2d(2*dim_f, 4*dim_f, 3, 2, 1),nn.PReLU(),nn.BatchNorm2d(4*dim_f),)
        self.conv4 = nn.Sequential(nn.Conv2d(4*dim_f, 8*dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(8*dim_f),)  
        self.conv5 = nn.Conv2d(8*dim_f, dim_z, 2, 1, 0) # 4,4 -->100 1 1 
                     
        
    def forward(self, inputs):   
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)   
        conv4 = self.conv4(conv3)      
        out = self.conv5(conv4)
        return out  
        

class Decoder(nn.Module):
    def __init__(self, dim_z=256, dim_c=3, dim_f=64):
        super(Decoder, self).__init__()        
        self.convT1 = nn.ConvTranspose2d(dim_z, 8*dim_f, 2, 1, 0) #  1,1 -->4,4 
        self.convT2 = nn.Sequential(nn.ConvTranspose2d(8*dim_f, 4*dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(4*dim_f),) 
        self.convT3 = nn.Sequential(nn.ConvTranspose2d(4*dim_f, 2*dim_f, 3, 2, 1),nn.PReLU(),nn.BatchNorm2d(2*dim_f),) 
        self.convT4 = nn.Sequential(nn.ConvTranspose2d(2*dim_f, dim_f, 4, 2, 1),nn.PReLU(),nn.BatchNorm2d(dim_f),) 
        self.convT5 = nn.Sequential(nn.ConvTranspose2d(dim_f, 1, 4, 2, 1),)
    
        
    def forward(self, inputs):
        convT1 = self.convT1(inputs)                                        
        convT2 = self.convT2(convT1)  
        convT3 = self.convT3(convT2) 
        convT4 = self.convT4(convT3) 
        output = self.convT5(convT4) 
        return output


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
        return decoded, encoded