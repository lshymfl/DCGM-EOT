import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn import init

def initialize_linear_weights(linear_layer, scale=1):
    init.kaiming_normal_(linear_layer.weight, a=0, mode='fan_in')
    linear_layer.weight.data *= scale
    if linear_layer.bias is not None:
        linear_layer.bias.data.zero_()


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
                init.constant_(m.bias.data, 0.1)


def get_scheduler(iters, optim):
    lr_lambda = lambda x : 1 - min((0.1*x / float(iters), 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    return scheduler

def init_optimizer(net, lr, iterations):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = get_scheduler(iterations, optimizer)
    return optimizer, scheduler


##############################################################################
class DualInputNet1(nn.Module):
    def __init__(self, input_feature_dim, input_info_dim):
        super(DualInputNet1, self).__init__()
        #self.feature_fc = nn.Linear(input_feature_dim, 128)  # 处理特征数据的全连接层
        self.feature_fc = nn.Linear(input_feature_dim, input_feature_dim)  # 处理特征数据的全连接层
        self.info_fc = nn.Linear(input_info_dim, 128)  # 处理额外信息的全连接层
        '''
        self.fc1 = nn.Linear(256, 128)
        
        self.fc2 = nn.Linear(128, 64)  # 合并特征和额外信息后的全连接层
        self.fc3 = nn.Linear(64, 1)  # 输出层，返回一个值
        '''
        
        sequence = [nn.Linear(input_feature_dim+128,512),
                        nn.BatchNorm1d(512),
                        nn.ReLU()]
        sequence += [
                nn.Linear(512,1024),
                nn.BatchNorm1d(1024),
                nn.ReLU()] #try to compare with: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        sequence += [
                nn.Linear(1024,512),
                nn.BatchNorm1d(512),
                nn.ReLU()]   
        sequence += [
                nn.Linear(512,256),
                nn.BatchNorm1d(256),
                nn.ReLU()]
        sequence += [
                nn.Linear(256,1)
                ]
        self.model = nn.Sequential(*sequence)

    def forward(self, feature_input, info_input):
        feature_output = torch.relu(self.feature_fc(feature_input))
        info_output = torch.relu(self.info_fc(info_input))
        
        # 合并两个输入的输出
        combined_output = torch.cat((feature_output, info_output), dim=1)
        '''
        x = torch.relu(self.fc1(combined_output))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        '''
        x = self.model(combined_output)
        x = x - torch.mean(x)
        return x
        
##############################################################################
##############################################################################
class DualInputNet(nn.Module):
    def __init__(self, input_feature_dim, input_info_dim):
        super(DualInputNet, self).__init__()
        self.feature_fc = nn.Linear(input_feature_dim, 1024)  # 处理特征数据的全连接层
        self.info_fc = nn.Linear(input_info_dim, 1024)  # 处理额外信息的全连接层
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)  # 合并特征和额外信息后的全连接层  
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        #self.dropout = nn.Dropout(0.1)  # 添加Dropout层
         

    def forward(self, feature_input, info_input):
        feature_output = torch.relu(self.feature_fc(feature_input))
        info_output = torch.relu(self.info_fc(info_input))
        
        # 合并两个输入的输出
        combined_output = torch.cat((feature_output, info_output), dim=1)
        
        x = torch.relu(self.fc1(combined_output))
        #x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        #x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        #x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        
        out = x.view(-1)
        out = out - torch.mean(out)
        return out
##############################################################################
##############################################################################
class DualInputNet2(nn.Module):
    def __init__(self, input_feature_dim, input_info_dim):
        super(DualInputNet2, self).__init__()
        self.feature_fc = nn.Linear(input_feature_dim, 1024)  # 处理特征数据的全连接层
        self.info_fc = nn.Linear(input_info_dim, 1024)  # 处理额外信息的全连接层
        #self.fc1 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)  # 合并特征和额外信息后的全连接层  
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        #self.dropout = nn.Dropout(0.1)  # 添加Dropout层
         

    def forward(self, feature_input, info_input):
        feature_output = torch.relu(self.feature_fc(feature_input))
        info_output = torch.relu(self.info_fc(info_input))
        
        # 合并两个输入的输出
        combined_output = torch.cat((feature_output, info_output), dim=1)
        
        x = torch.relu(self.fc1(combined_output))
        #x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        #x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        #x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        #x = torch.relu(self.fc6(x))
        x = self.fc6(x)
        
        out = x.view(-1)
        out = out - torch.mean(out)
        return out
                
        
        
        
        
        
        