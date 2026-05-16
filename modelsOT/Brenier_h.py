# -*- coding: utf-8 -*-
# @Time        : 16/12/2021 17:10 PM
# @Description :
# @Author      : li zezeng
# @Email       : zezeng.lee@gmail.com

import torch.nn as nn
import torch.nn.init as init
import torch
import numpy as np
import torch.nn.functional as F

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class MLP_Attention(nn.Module):

    def __init__(self, d_model,n=49,temperature=1, attn_dropout=0.1,simple=False):

        super(MLP_Attention, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model,d_model)
        if(simple):
            self.position_biases=torch.zeros((n,n))
        else:
            self.position_biases=nn.Parameter(torch.ones((n,n)))
        self.d_model = d_model
        self.n=n
        self.sigmoid=nn.Sigmoid()
        
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        bs, n,dim = input.shape

        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input).view(1,bs,n,dim) #1,bs,n,dim
        v = self.fc_v(input).view(1,bs,n,dim) #1,bs,n,dim
        
        numerator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1))*v,dim=2) #n,bs,dim
        denominator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1)),dim=2) #n,bs,dim

        out=(numerator/denominator) #n,bs,dim
        out=self.sigmoid(q)*(out.permute(1,0,2)) #bs,n,dim'''
        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input)
        v = self.fc_v(input)
        attn = torch.matmul(q / self.temperature, k.transpose(1, 0))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        

        return out
        

class MyReLU(torch.autograd.Function):
    #@staticmethod
    #def forward(ctx, input):
        #ctx.save_for_backward(input, weight, bias)
    def forward(self, input):
        self.save_for_backward(input)
        output = input_.clamp(min=0) 
        return output

    #@staticmethod
    #def backward(ctx, grad_output):
        #input, weight, bias = ctx.saved_variables
    def backward(self, grad_output):
        input_, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0             
        return grad_input


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.uniform_(-0.1, 0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)



class Brenier_h(nn.Module):
    def __init__(self, in_dim, use_bn=True,first_omega_0=3000):
        super(Brenier_h,self).__init__()
        if use_bn:
            sequence = [nn.Linear(in_dim,512),
                        nn.BatchNorm1d(512),
                        nn.ReLU()]
            sequence += [
                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.ReLU()] #try to compare with: nn.LeakyReLU(negative_slope=0.2, inplace=True)
            sequence += [
                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.ReLU()]
            '''    
            sequence += [
                nn.Linear(512,512),
                nn.BatchNorm1d(512),
                nn.ReLU()]'''
                
            sequence += [
                nn.Linear(512,1)
                ]
        else:
            '''sequence = [nn.Linear(in_dim,512),nn.ReLU()]
            sequence += [nn.Linear(512,512),nn.ReLU()] 
            sequence += [nn.Linear(512,512),nn.ReLU()]
            sequence += [nn.Linear(512,1)]
            '''
            
            sequence = [nn.Linear(in_dim,512),
                        SineLayer(512, 512, is_first=True, omega_0=first_omega_0)]
            sequence += [nn.Linear(512,512),
                         SineLayer(512, 512, is_first=False, omega_0=first_omega_0)] 
            sequence += [nn.Linear(512,512),
                         SineLayer(512, 512, is_first=False, omega_0=first_omega_0)] 
            sequence += [nn.Linear(512,1)]
            '''
            sequence = [nn.Linear(in_dim,512),nn.ReLU()]
            sequence += [MLP_Attention(512,512),nn.ReLU()] 
            sequence += [MLP_Attention(512,512),nn.ReLU()] 
            sequence += [MLP_Attention(512,512),nn.ReLU()] 
            sequence += [nn.Linear(512,1)]'''
            

        self.model = nn.Sequential(*sequence)
        initialize_weights(self.model)

    def forward(self,input):
        out = self.model(input)
        res =out-torch.mean(out)
        return res


