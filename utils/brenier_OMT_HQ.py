import sys
import os
import torch
import numpy as np
from modelsOT.Brenier_h import Brenier_h
from torch.optim.lr_scheduler import MultiStepLR

torch.set_printoptions(precision=8)
class OMT():	
    '''This class is designed to compute the semi-discrete Optimal Transport (OT) problem. 
    Specifically, within the unit cube [0,1]^n of the n-dim Euclidean space,
    given a source continuous distribution mu, and a discrete target distribution nu = \sum nu_i * \delta(P_i),
    where \delta(x) is the Dirac function at x \in [0,1]^n, compute the Optimal Transport map pushing forward mu to nu.

    The method is based on the variational principle of solving semi-discrete OT, (See e.g.
    Gu, Xianfeng, et al. "Variational principles for Minkowski type problems, discrete optimal transport, and discrete Monge-Ampere equations." Asian Journal of Mathematics 20.2 (2016): 383-398.)
    where a convex energy is minimized to obtain the OT map. 

    Adam gradient descent method is used here to perform the optimization, and Monte-Carlo integration method is used to calculate the energy.
    '''

    def __init__ (self, h_P, num_P, dim, max_iter, lr, bat_size_P, bat_size_n, eps, randmeasure, out_dir='./FFHQ_new/ot_models/'):
        '''Parameters to compute semi-discrete Optimal Transport (OT)
        Args:
            h_P: Host vector (i.e. CPU vector) storing locations of target points with float type and of shape (num_P, dim).
            num_P: A positive interger indicating the number of target points (i.e. points the target discrete measure concentrates on).
            dim: A positive integer indicating the ambient dimension of OT problem.
            max_iter: A positive integer indicating the maximum steps the gradient descent would iterate.
            lr: A positive float number indicating the step length (i.e. learning rate) of the gradient descent algorithm.
            bat_size_P: Size of mini-batch of h_P that feeds to device (i.e. GPU). Positive integer.
            bat_size_n: Size of mini-batch of Monte-Carlo samples on device. The total number of MC samples used in each iteration is batch_size_n * num_bat.
        '''
        self.h_P = h_P.cuda()
        #self.dim1, self.dim2, self.dim3 = h_P.shape[1], h_P.shape[2], h_P.shape[3]
        self.num_P = num_P
        self.dim = dim
        self.max_iter = max_iter
        self.lr = lr
        self.bat_size_P = bat_size_P
        self.bat_size_n = bat_size_n
        self.eps = eps
        #self.feature_indices = feature_indices
        self.randmeasure = randmeasure
           
        if num_P % bat_size_P != 0:
        	sys.exit('Error: (num_P) is not a multiple of (bat_size_P)')
        if num_P > 200000:
            #self.bat_size_P = 56802
            self.num_bat_P = num_P // self.bat_size_P
        else:
            self.num_bat_P = num_P // bat_size_P 
        print(self.num_bat_P)
        
        self.epochs_per_save = 1
        self.out_dir = out_dir

        #self.num_bat_P = num_P // bat_size_P
        #!internal variables
        '''
        self.d_volP: Generated mini-batch of MC samples on device (i.e. GPU) of shape (self.bat_size_n, dim).
        self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        self.d_g: The gradient of the energy function E(h).
        self.d_U: Convex envelope of all piecewise linear functions.
        self.d_ind: Monte Carlo small batch sampling falls at the corresponding target point index.
        self.d_tot_ind: The index where all sampling points fall on the corresponding target point.
        self.d_adam_m: First order momentum parameter.
        self.d_adam_v: Second order momentum parameter.
        '''
        #self.d_G_z = torch.empty(self.bat_size_n*self.dim, dtype=torch.float, device=torch.device('cuda'))
        self.d_volP = torch.empty((self.bat_size_n, self.dim), dtype=torch.float, device=torch.device('cuda'))
        self.d_h = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        #self.d_h = torch.zeros(self.bat_size_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_delta_h = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_ind = torch.empty(self.bat_size_n, dtype=torch.long, device=torch.device('cuda'))
        self.d_ind_val = torch.empty(self.bat_size_n, dtype=torch.float, device=torch.device('cuda'))
        
        self.d_ind_val_argmax = torch.empty(self.bat_size_n, dtype=torch.long, device=torch.device('cuda'))
        self.d_tot_ind = torch.empty(self.bat_size_n, dtype=torch.long, device=torch.device('cuda'))
        self.d_tot_ind_val = torch.empty(self.bat_size_n, dtype=torch.float, device=torch.device('cuda'))
        self.d_g = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_g_sum = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_adam_m = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_adam_v = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))

        #!temp variables
        self.d_U = torch.empty((self.bat_size_P, self.bat_size_n), dtype=torch.float, device=torch.device('cuda'))
        self.d_temp_h = torch.empty(self.bat_size_P, dtype=torch.float, device=torch.device('cuda'))
        self.d_temp_P = torch.empty((self.bat_size_P, self.dim), dtype=torch.float, device=torch.device('cuda'))
        #self.d_temp_P = torch.empty((self.bat_size_P, self.dim1, self.dim2, self.dim3), dtype=torch.float, device=torch.device('cuda'))
        
        ###!random number generator  torch.rand() and torch.randn() are achieved same effect
        #self.qrng = torch.quasirandom.SobolEngine(dimension=self.dim)  # celeba
        
        if self.randmeasure:
            total = torch.randint(0, self.num_P, (self.num_P,))
            self.target_prob = torch.div( total, sum(total) ).cuda()
        else:
            self.target_prob = torch.div(torch.ones(self.num_P).cuda(), self.num_P)
        print(self.target_prob)
        
        self.brenier_h = Brenier_h(self.dim)
        self.brenier_h.to(torch.device('cuda'))
        self.optimizer = torch.optim.Adam(self.brenier_h.parameters(),lr=self.lr, weight_decay=2e-4)#
        self.scheduler = MultiStepLR(self.optimizer, milestones=[3000,6000], gamma=0.1)#log3.out
        #self.scheduler = MultiStepLR(self.optimizer, milestones=[4000,7000,10000,13000,16000], gamma=0.25)
        self.loss_fn = torch.nn.MSELoss()
        self.loss_fn.to(torch.device('cuda'))
        self.start_epoch = 0
         
        #self.qrng = torch.randn((self.bat_size_n*self.bat_size_n, self.dim), dtype=torch.float).cuda()
  
        print('Allocated GPU memory: {}MB'.format(torch.cuda.memory_allocated()/1e6))
        print('Cached memory: {}MB'.format(torch.cuda.memory_cached()/1e6))
  
    def gaussian_kernel(x, variance):
        #  
        return torch.exp(-torch.norm(x, dim=1)**2 / (2 * variance))
   
    def pre_cal(self,count):
        '''Monte-Carlo sample generator.
        Args: count: Index of MC mini-batch to generate in the current iteration step. Used to set the state of random number generator.
        Returns: self.d_volP: Generated mini-batch of MC samples on device (i.e. GPU) of shape (self.bat_size_n, dim).
        '''
        #self.d_volP = torch.randn((self.bat_size_n, self.dim), dtype=torch.float).cuda()
        self.d_volP = torch.randn((self.bat_size_n, self.dim), dtype=torch.float).cuda()  ## used randn
        #self.d_volP = torch.randn((self.bat_size_n, self.dim1, self.dim2, self.dim3), dtype=torch.float).cuda()  
        
         
        #self.qrng.draw(self.bat_size_n,out=self.d_volP)  # celeba
        #self.d_volP.add_(-0.5)
        
    def cal_measure(self):
        '''Calculate the pushed-forward measure of current step. 
        '''
        self.d_tot_ind_val.fill_(-1e30)
        self.d_tot_ind.fill_(-1)
        i = 0     
        while i < self.num_bat_P:
            temp_P = self.h_P[i*self.bat_size_P:(i+1)*self.bat_size_P]
            temp_P = temp_P.view(temp_P.shape[0], -1) 
                
            '''U=PX+H'''
            d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
            self.d_U = torch.mm(temp_P, self.d_volP.t())+d_temp_h.expand([self.bat_size_n, -1]).t()

            '''compute max'''
            d_ind_val, d_ind = torch.max(self.d_U, 0)
            '''add P id offset'''
            d_ind = d_ind+(i*self.bat_size_P)
            '''store best value'''
            self.d_tot_ind_val, d_ind_val_argmax = torch.max(torch.stack((self.d_tot_ind_val, d_ind_val)), 0)
            self.d_tot_ind = torch.stack((self.d_tot_ind, d_ind))[d_ind_val_argmax, torch.arange(self.bat_size_n)] 
            '''add step'''
            i = i+1
     
        '''calculate histogram'''
        self.d_g = torch.bincount(self.d_tot_ind, minlength=self.num_P)
        self.d_g = self.d_g /self.bat_size_n
    
    def train_ot(self, num_bat=1,ckpt_path=None):
        '''Gradient descent method. Update self.d_h to the optimal solution.
        Args:
            last_step: Iteration performed before the calling. Used when resuming the training. Default [0].
            num_bat: Starting number of mini-batch of Monte-Carlo samples. Value of num_bat will increase during iteration. Default [1].
                     total number of MC samples used in each iteration = self.batch_size_n * num_bat
        Returns:
            self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        '''
        if ckpt_path is not None:
            ckpt = self._get_latest_checkpoint(ckpt_path)
            if ckpt is not None:
                self._load_checkpoint(ckpt=ckpt)
        last_step = self.start_epoch


        best_g_norm = 1e20
        curr_best_g_norm = 1e20
        steps = 0
        count_bad = 0
        dyn_num_bat_n = num_bat
        h_file_list = []
        
        d_adam_m = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        d_adam_v = torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))

        self.brenier_h.train()
        previ_h = 0.001*torch.ones(self.num_P, dtype=torch.float, device=torch.device('cuda'))
        while(steps <= self.max_iter):
            #self.qrng.reset()
            d_g_sum=torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
            for count in range(dyn_num_bat_n):               
                self.pre_cal(count)
                self.cal_measure()
                d_g_sum = d_g_sum + self.d_g

            self.d_g = d_g_sum/dyn_num_bat_n			
            pred_h = self.brenier_h(self.h_P).squeeze()
            self.optimizer.zero_grad()
            bias_grd = (self.d_g - self.target_prob).detach()
            pred_h.backward(bias_grd)
            
            #adam  will lead to the loss that cannot be reduced
            '''d_adam_m *= 0.9
            d_adam_m += 0.1*bias_grd
            d_adam_v *= 0.9999
            d_adam_v += 0.0001*bias_grd*bias_grd
            grad = torch.div(d_adam_m, torch.add(torch.sqrt(d_adam_v),1e-8)).detach()
            pred_h.backward(grad)'''
            
            #fixed point method will lead to the loss that cannot be reduced
            '''
            diff = pred_h-previ_h
            loss = torch.sum(torch.abs(bias_grd).detach()*diff*diff)
            loss.backward()
            previ_h = pred_h.clone().detach()'''
            
            
            self.optimizer.step()
            self.scheduler.step()
            
            self.d_h.copy_(pred_h)
            
            g_norm = torch.sqrt(torch.sum(torch.mul(bias_grd,bias_grd)))
            
            
            loss = self.loss_fn(self.d_g,self.target_prob)
            print('loss=',loss)  
            num_zero = torch.sum(bias_grd == -1./self.num_P)
            ratio_diff = torch.max(bias_grd)   
            print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g_norm: {3:.6f}. num_zero: {4:d}'.format(
                    steps, self.max_iter,ratio_diff, g_norm, num_zero))
            '''
            if (steps+1) % 100 == 0:
                loss = self.loss_fn(self.d_g,self.target_prob)
                print('loss=',loss)      
                num_zero = torch.sum(bias_grd == -1./self.num_P)
                ratio_diff = torch.max(bias_grd)   
                print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
                    steps, self.max_iter, ratio_diff, g_norm, num_zero))
            '''
                
                #for name, parms in self.brenier_h.named_parameters(): 
                    #print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)
            
            model_name = '{}-{:0>4d}.pt'.format('OTnet', steps)
            model_name = os.path.join(self.out_dir, model_name)
            is_converged = (g_norm < self.eps)
            if (steps+1)%10==0 or is_converged:
                self._save_checkpoint({
                    'epoch': steps,
                    'state_dict': self.brenier_h.state_dict()}, model_name)

            if g_norm < self.eps:
                return       
             
            
            '''
            if (steps+1) % self.epochs_per_save == 0 or steps+1 == self.max_iter:
                model_name = '{}-{:0>6d}.pt'.format('OTnet', steps+1)
                model_name = os.path.join(self.out_dir, model_name)
                self._save_checkpoint({
                    'epoch': steps,
                    'state_dict': self.brenier_h.state_dict()}, model_name)
            '''
            
             

            if g_norm <= curr_best_g_norm:
                curr_best_g_norm = g_norm
                count_bad = 0
            else:
                count_bad += 1
            if count_bad > 30 and num_bat*8>dyn_num_bat_n:
                dyn_num_bat_n *= 2
                print('bat_size_n has increased to {}'.format(dyn_num_bat_n*self.bat_size_n))
                count_bad = 0
                curr_best_g_norm = 1e20

            steps += 1


    def set_h(self, h_tensor):
        self.d_h.copy_(h_tensor).to(self.device)
        pass
    
    
    def get_h(self, target, model_file_name= None):
        #self.brenier_h.eval()
        if model_file_name is not None:
            self._load_checkpoint(self, ckpt)
        with torch.no_grad():
            pred = self.brenier_h(target).squeeze()
        return pred
        
    def save_h(self, output_h_file):
        #self.brenier_h.eval()
        with torch.no_grad():
            pred = self.brenier_h(self.h_P).squeeze()
            torch.save(pred, output_h_file)        

    @staticmethod
    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.brenier_h.load_state_dict(checkpoint['state_dict'])
            self.start_epoch = checkpoint['epoch']+1
            print("[*] loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        if len(ckpts)>=1:
            all_times = sorted(ckpts, reverse=True)
            return os.path.join(path, all_times[0])
        else:
            return None

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        
        