import sys
import os
import torch
import numpy as np

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

    def __init__ (self, h_P, num_P, dim, max_iter, lr_ot, bat_size_P, bat_size_n, eps, alter_measture, alter_target_measure):
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
        self.num_P = num_P
        self.dim = dim
        self.max_iter = max_iter
        self.lr = lr_ot
        self.bat_size_P = bat_size_P
        self.bat_size_n = bat_size_n
        self.eps = eps
        self.alter_measture = alter_measture
        self.alter_target_measure = alter_target_measure

        #self.feature_indices = feature_indices
           
        if num_P % bat_size_P != 0:
        	sys.exit('Error: (num_P) is not a multiple of (bat_size_P)')
        if num_P > 200000:
            self.bat_size_P = 100000
            self.num_bat_P = num_P // self.bat_size_P
        else:
            self.num_bat_P = num_P // bat_size_P 
        print(self.num_bat_P)

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
        
        if self.alter_measture:
            self.target_measure = self.alter_target_measure.cuda()
        else:
            #total = torch.rand(self.num_P)  
            #threshold = 0.2   
            #total[total < threshold] = 0
            #self.target_measure = torch.div( total, sum(total) ).cuda()
            ####self.target_measure = torch.div(torch.ones(self.num_P), self.num_P).cuda()
            class_num1 = [2130, 21456, 105, 1517, 6830, 44625, 403, 21332, 364, 26126, 1536, 59808]
            class_num = sorted(class_num1, reverse=True)
            total_num = sum(class_num)  

            class_num = torch.tensor(class_num, dtype=torch.float64)
            class_ratio = class_num / total_num  
            
            target_measure = []
            for i in range(class_num.shape[0]):
                measure = torch.ones(int(class_num[i]))*class_ratio[i]
                target_measure.append(measure)  
            self.target_measure = torch.cat(target_measure)/torch.sum(torch.cat(target_measure))  
            self.target_measure = self.target_measure.cuda()
            
        print('self.target_measure=',self.target_measure, torch.sum(self.target_measure)) 
        
        ###!random number generator  torch.rand() and torch.randn() are achieved same effect
        #self.qrng = torch.quasirandom.SobolEngine(dimension=self.dim)  # celeba
  
        print('Allocated GPU memory: {}MB'.format(torch.cuda.memory_allocated()/1e6))
        print('Cached memory: {}MB'.format(torch.cuda.memory_cached()/1e6))
  
    
    def pre_cal(self,count):
        '''Monte-Carlo sample generator.
        Args: count: Index of MC mini-batch to generate in the current iteration step. Used to set the state of random number generator.
        Returns: self.d_volP: Generated mini-batch of MC samples on device (i.e. GPU) of shape (self.bat_size_n, dim).
        '''
        self.d_volP = torch.rand((self.bat_size_n, self.dim), dtype=torch.float).cuda()  ## used randn
        
        #self.qrng.draw(self.bat_size_n,out=self.d_volP)  # celeba
        #self.d_volP.add_(-0.5)
        
    def cal_measure(self):
        '''Calculate the pushed-forward measure of current step. 
        '''
        self.d_tot_ind_val.fill_(-1e30)
        self.d_tot_ind.fill_(-1)
        i = 0     
        while i < self.num_bat_P:  #self.num_P // self.bat_size_P:
            temp_P = self.h_P[i*self.bat_size_P:(i+1)*self.bat_size_P]
            temp_P = temp_P.view(temp_P.shape[0], -1)	
                
            '''U=PX+H'''
            self.d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
            self.d_temp_P.copy_(temp_P)
            torch.mm(self.d_temp_P, self.d_volP.t(),out=self.d_U)
            torch.add(self.d_U, self.d_temp_h.expand([self.bat_size_n, -1]).t(), out=self.d_U)

            '''compute max'''
            torch.max(self.d_U, 0, out=(self.d_ind_val, self.d_ind))
            '''add P id offset'''
            self.d_ind.add_(i*self.bat_size_P)
            '''store best value'''
            torch.max(torch.stack((self.d_tot_ind_val, self.d_ind_val)), 0, out=(self.d_tot_ind_val, self.d_ind_val_argmax))
            self.d_tot_ind = torch.stack((self.d_tot_ind, self.d_ind))[self.d_ind_val_argmax, torch.arange(self.bat_size_n)]
          
            '''add step'''
            i = i+1
     
        '''calculate histogram'''
        self.d_g.copy_(torch.bincount(self.d_tot_ind, minlength=self.num_P))        
        self.d_g.div_(self.bat_size_n)

    def update_h(self):
        '''Calculate the update step based on gradient'''
        #self.d_g -= 1./self.num_P            ###################modified
        self.d_g = self.d_g - self.target_measure           ###################modified
        self.d_adam_m *= 0.9
        self.d_adam_m += 0.1*self.d_g
        self.d_adam_v *= 0.999
        self.d_adam_v += 0.001*torch.mul(self.d_g,self.d_g)
        torch.mul(torch.div(self.d_adam_m, torch.add(torch.sqrt(self.d_adam_v),1e-8)),-self.lr,out=self.d_delta_h)
        torch.add(self.d_h, self.d_delta_h, out=self.d_h)
        '''normalize h'''
        self.d_h -= torch.mean(self.d_h)

    def run_gd(self, last_step=0, num_bat=1):
        '''Gradient descent method. Update self.d_h to the optimal solution.
        Args:
            last_step: Iteration performed before the calling. Used when resuming the training. Default [0].
            num_bat: Starting number of mini-batch of Monte-Carlo samples. Value of num_bat will increase during iteration. Default [1].
                     total number of MC samples used in each iteration = self.batch_size_n * num_bat
        Returns:
            self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        '''
        g_ratio = 1e20
        best_g_norm = 1e20
        curr_best_g_norm = 1e20
        steps = 0
        count_bad = 0
        dyn_num_bat_n = num_bat
        h_file_list = []
        m_file_list = []
        v_file_list = []
        print(dyn_num_bat_n)
        while(steps < self.max_iter):
            #self.qrng.reset()
            #self.d_g_sum.fill_(0.)
            self.d_g_sum=torch.zeros(self.num_P, dtype=torch.float, device=torch.device('cuda'))
            for count in range(dyn_num_bat_n):               
                self.pre_cal(count)
                #torch.save(self.d_volP, './AE-all-2/source_cube_point/{:08d}.pt'.format((count+1+steps*dyn_num_bat_n))) # load data
                self.cal_measure()
                #torch.save(self.d_tot_ind, './AE-all-2/source_cube_index/{:08d}.pt'.format((count+1+steps*dyn_num_bat_n))) # load index
                
                self.d_g_sum = self.d_g_sum + self.d_g
            self.d_g = self.d_g_sum/dyn_num_bat_n		
            self.update_h()
            
            g_norm = torch.sqrt(torch.sum(torch.mul(self.d_g,self.d_g)))
            #num_zero = torch.sum(self.d_g == -1./self.num_P)
            num_zero = torch.sum(self.d_g == -self.target_measure)  ## modified
            torch.abs(self.d_g, out=self.d_g)
            g_ratio = torch.max(self.d_g)*self.num_P            
            print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
                steps, self.max_iter, g_ratio, g_norm, num_zero))

            if g_norm < self.eps:
                return       

            if g_norm <= curr_best_g_norm:
                curr_best_g_norm = g_norm
                count_bad = 0
            else:
                count_bad += 1
            if count_bad > 10:
                dyn_num_bat_n *= 2
                print('bat_size_n has increased to {}'.format(dyn_num_bat_n*self.bat_size_n))
                count_bad = 0
                curr_best_g_norm = 1e20

            steps += 1

    def set_h(self, h_tensor):
        #print('h_tensor=', h_tensor.shape)
        if h_tensor.shape[0] > 200000:
            print('h_tensor=', h_tensor.shape)
            #h_tensor = h_tensor[self.feature_indices]
            #self.d_h.copy_(h_tensor.detach())
            self.d_h.copy_(h_tensor)
        else:
            self.d_h.copy_(h_tensor.detach())
            #self.d_h.copy_(h_tensor)
            #self.d_h = (self.d_h - torch.min(self.d_h)) / (torch.max(self.d_h) - torch.min(self.d_h))

def train_omt(p_s, num_bat=1):
    last_step = 0
    
    '''run gradient descent'''
    p_s.run_gd(last_step=last_step, num_bat=num_bat)

def tuning_omt(p_s, selected_ot_model_path, num_bat=1):
    last_step = 0
    p_s.set_h(torch.load(selected_ot_model_path))
    
    '''run gradient descent'''
    p_s.run_gd(last_step=last_step, num_bat=num_bat)
