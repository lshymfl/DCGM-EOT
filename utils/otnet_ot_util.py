import torch
import numpy as np
import scipy.io as sio
import os
from utils.brenier_OMT_HQ import OMT 
import time
#from models.Brenier_u_u import OptimalTransport

torch.set_printoptions(precision=8)

#  generate latent code P
def gen_P(OTNet, numX, feature_indices, args):
    topk = args.top_k
    I_all = -torch.ones([topk, numX], dtype=torch.long)
    num_bat_x = numX//OTNet.bat_size_n
    bat_size_x = min(numX, OTNet.bat_size_n)
    for ii in range(max(num_bat_x, 1)):
        OTNet.pre_cal(ii)
        OTNet.cal_measure()
        _, I = torch.topk(OTNet.d_U, topk, dim=0)
        for k in range(topk):
            I_all[k, ii*bat_size_x:(ii+1)*bat_size_x].copy_(I[k, 0:bat_size_x])
    I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long)
    for ii in range(topk-1):
        I_all_2[0, ii * numX:(ii+1) * numX] = I_all[0,:]
        I_all_2[1, ii * numX:(ii+1) * numX] = I_all[ii + 1, :]
    I_all = I_all_2
   
    if torch.sum(I_all < 0) > 0:
        print('Error: numX is not a multiple of bat_size_n')

    ###compute angles
    P = (OTNet.h_P).cpu() 
    if P.shape[0] > 200000:
        P = P[feature_indices]
        #P = P[0:100000]
    else:
        P = (OTNet.h_P).cpu()   
    nm = torch.cat([P, -torch.ones(OTNet.num_P,1)], dim=1)
    nm /= torch.norm(nm,dim=1).view(-1,1)
    cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
    cs = torch.min(torch.ones([cs.shape[0]]), cs)
    theta = torch.acos(cs)
    print('theta_max=',torch.max(theta))
    theta = (theta-torch.min(theta))/(torch.max(theta)-torch.min(theta))

    ###filter out generated samples with theta larger than threshold
    I_gen = I_all[:, theta <= args.angle_thresh]
    I_gen, _ = torch.sort(I_gen, dim=0)
    _, uni_gen_id = np.unique(I_gen[0,:].numpy(), return_index=True)
    np.random.shuffle(uni_gen_id)
    I_gen = I_gen[:, torch.from_numpy(uni_gen_id)]
     
    numGen = I_gen.shape[1]
    if args.gen_nums is not None:
        numGen = min(numGen, args.gen_nums)
    I_gen = I_gen[:,:numGen]
    print('OT successfully generated {} samples'.format(numGen))
    
    ###generate new features
    P_gen2 = P[I_gen[0,:],:]
    ###the first way   
    rand_w = args.dissim * torch.ones([numGen,1])
    P_gen = (torch.mul(P[I_gen[0,:],:], 1 - rand_w) + torch.mul(P[I_gen[1,:],:], rand_w)).numpy()
    #P_gen = np.concatenate((P_gen,P_gen2))
    
    id_gen = I_gen[0,:].squeeze().numpy().astype(int)
    gen_feature_path = os.path.join(args.result_root_path,'gen_features.mat')
    sio.savemat(gen_feature_path, {'features':P_gen, 'ids':id_gen})

def compute_otnet(ae_feature_path, args, mode='train'):
    #######arguments for training OT
    h_P_z = torch.load(ae_feature_path)  # AE encoder latent space features
    h_P = h_P_z.view(h_P_z.shape[0], -1)
    print("h_P_z=", h_P_z.shape, "h_P=", h_P.shape)
    #h_P = torch.load(ae_feature_path)  # AE encoder latent space features
    #print("h_P=", h_P.shape)

    #torch.manual_seed(1234)
    #feature_indices = torch.randperm(h_P.size(0))
    
    if h_P.shape[0] > 200000:
        #h_P = h_P[feature_indices]
        num_P = h_P.shape[0]
        bat_size_P = 76813
    else:
        num_P = h_P.shape[0]
        bat_size_P = num_P
    
    dim_y = h_P.shape[1]
    maxIter = args.maxIterotnet
    lr_ot = args.lr_otnet
    bat_size_n = args.bat_size_n
    init_num_bat_n = args.init_num_bat_n
    num_gen_x = bat_size_n*init_num_bat_n     #a multiple of bat_size_n
    eps = args.eps
    N = args.N
    randmeasure = args.randmeasure 
    
    #h_P_norm = h_P_norm[0:num_P//bat_size_P*bat_size_P,:]##add
    #num_P = h_P.shape[0]

    #p_s = OMT(h_P, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps, feature_indices)
    #p_s = OMT(h_P_norm, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps)###modified
   
    out_dir = os.path.join(args.result_root_path,'ot_models')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #OTNet = OMT(h_P, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps, feature_indices, out_dir=out_dir)
  
    
    '''train omt'''
    if mode=='train':
        
        h_sample = []
        nu_sample = []
        Y_sample = torch.empty([0,h_P.size(1)])
        for i in range(N):
            print('N=', i, N)
            OTNet = OMT(h_P, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps, randmeasure, out_dir=out_dir)
            OTNet.train_ot(init_num_bat_n)
            pred = torch.zeros(num_P)
            batch_n = num_P//args.batch_size
            for i in range(batch_n):
                pred[i*args.batch_size:(i+1)*args.batch_size]=OTNet.get_h(h_P[i*args.batch_size:(i+1)*args.batch_size,:].cuda()).cpu() 
            remain = num_P-args.batch_size*batch_n
            if remain>0:
                pred[args.batch_size*batch_n:num_P]=OTNet.get_h(h_P[args.batch_size*batch_n:num_P,:].cuda()).cpu()
            
            h = OTNet.d_h 
            h_sample.append(h)
            nu = OTNet.target_prob
            nu_sample.append(nu)
            Y_sample = torch.cat( (Y_sample,h_P) ,dim=0)
        h_sample = torch.cat(h_sample, dim=0) 
        nu_sample = torch.cat(nu_sample, dim=0)
        print(h_sample.shape, nu_sample.shape)
        print('h=', h_sample)
        print('nu=', nu_sample)
        torch.save(h_sample, os.path.join(args.result_root_path, 'h_sample.pt') )
        torch.save(nu_sample, os.path.join(args.result_root_path, 'nu_sample.pt') )
        torch.save(Y_sample, os.path.join(args.result_root_path, 'Y_sample.pt') )
    
    
  
