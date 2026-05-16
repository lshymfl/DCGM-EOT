import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import  math

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, mode='down'):
        assert mode in ['down', 'up'], "Mode must be either 'down' or 'up'."
        super(ResBlock, self).__init__()
        if mode == 'down':
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.activate = nn.LeakyReLU(0.2, inplace=True)
        elif mode == 'up':
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.activate = nn.ReLU(inplace=True)   ###nn.ReLU(inplace=True) 
        self.BN = nn.BatchNorm2d(out_channels)   ###nn.BatchNorm2d(c_out, momentum=0.1)
        self.resize = stride > 1 or (stride == 1 and padding == 0) or out_channels != in_channels
    
    def forward(self, x):
        conv1 = self.BN(self.conv1(x)) 
        relu = self.activate(conv1)       
        conv2 = self.BN(self.conv2(relu))
        if self.resize:           
            x = self.BN(self.conv1(x))
        return self.activate(x + conv2)


class AttentionBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(x)
        out = self.sigmoid(out)
        return out * x


class AttentionBlock2(nn.Module):
    def __init__(self, in_channels, gamma):
        super(AttentionBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.gamma = gamma #nn.Parameter(torch.tensor(1))

    def forward(self, x):
        f = self.conv1(x)      
        g = self.conv2(x)
        k = self.conv3(x)
        b, c, h, w = f.size()
        f = f.view(b, -1, h * w).permute(0, 2, 1)
        g = g.view(b, -1, h * w)
        k = k.view(b, -1, h * w).permute(0, 2, 1)
        attention_map = F.softmax(torch.bmm(f, g), dim=2)
        attention_map = torch.bmm(attention_map, k).permute(0, 2, 1).contiguous()
        attention_map = attention_map.view(b, c, h, w)
        out = self.gamma * attention_map + x
        #print(self.gamma)
        return out
 

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = vgg19(pretrained=True).cuda()
        self.features = nn.Sequential(*list(vgg.features.children())[:8])

    def forward(self, x):
        return self.features(x)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, target):
        input_features = self.feature_extractor(inputs)
        target_features = self.feature_extractor(target)
        loss = self.mse_loss(input_features, target_features)
        return loss


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=0.5, device='cuda'):
        self.window = self.create_window(window_size, sigma).to(device)
        self.window_size = window_size
        
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def create_window(self, window_size, sigma):
        _1D_window = self.gaussian(window_size, sigma)
        _2D_window = _1D_window.unsqueeze(1) * _1D_window.unsqueeze(0)
        window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
        return window
    
    def __call__(self, img1, img2, c1 = (0.001 * 255) ** 2, c2 = (0.003 * 255) ** 2):
        img1 = img1.permute(0, 3, 1, 2)
        img2 = img2.permute(0, 3, 1, 2)
        
        (_, channel, _, _) = img1.size()
        
        window = self.window
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = c1
        C2 = c2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_loss = ssim_map.mean()  #torch.mean((1 - ssim_map) / 2)
        
        return 1-ssim_loss

# difine learning weight of MSE and LPIPS 
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.alpha = nn.Parameter(torch.clamp(torch.tensor(0.1), 0, 1))
        #self.alpha = torch.clamp(self.alpha, 0, 1)
        
        
    def forward(self, mse_loss, lpips_loss):
        loss = (1 - self.alpha) * mse_loss  + self.alpha * lpips_loss 
        #print(self.alpha)
        return loss

    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        dist = torch.dist(x1, x2, p=2)
        loss = y * torch.pow(dist, 2) + (1 - y) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()

 


class GaborFilter(nn.Module):
    def __init__(self, num_channels, num_orientations, kernel_size, sigma, theta):
        super(GaborFilter, self).__init__()
        self.num_channels = num_channels
        self.num_orientations = num_orientations
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta
        self.filters = nn.Parameter(torch.randn(num_channels * num_orientations, kernel_size, kernel_size))
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        gabor_filters = self.filters.view(self.num_channels, self.num_orientations, self.kernel_size, self.kernel_size)
        responses = []
        for i in range(self.num_channels):
            for j in range(self.num_orientations):
                filter = gabor_filters[i, j, :, :].unsqueeze(0).unsqueeze(1)
                response = F.conv2d(x[:, i:i+1, :, :], filter, padding=self.kernel_size // 2)
                responses.append(response)
        return torch.cat(responses, dim=1)

############################
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class CorresDiffusionStep(nn.Module):
    def __init__(self, beta_1, beta_T, T):
        super().__init__()

        #self.model = model
        self.T = T
        #self.corresStep = corresStep
        
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double()) ## Determine/define a constant  beta = 0.0001->0.02
        alphas = 1. - self.betas   ##
        alphas_bar = torch.cumprod(alphas, dim=0)  ## = alpha_1*alpha_2*...*alpha_n

        # calculations for diffusion q(x_t | x_{t-1}) and others   = sqrt(sqrt_alphas_bar)*x_0 + sqrt(1-sqrt_alphas_bar)*z
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        #torch.manual_seed(seed) # to CPU 
        
        t = torch.randint(self.T-1, self.T, size=(x_0.shape[0], ) ) 
        #torch.cuda.manual_seed(1234) # to GPU 
        noise = torch.randn_like(x_0) 
        #print(noise[0])
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise) 
        return x_t
###################################