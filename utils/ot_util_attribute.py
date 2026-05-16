import torch
import numpy as np
import scipy.io as sio
import os
from utils.OMT_attribute import OMT, train_omt, tuning_omt
import time
import torch.nn.functional as F

torch.set_printoptions(precision=8)

#  generate latent code P
def gen_P(OTNet, numX, args):
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
        P = P

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
    if args.max_gen_samples is not None:
        numGen = min(numGen, args.max_gen_samples)
    I_gen = I_gen[:,:numGen]
    print('OT successfully generated {} samples'.format(
        numGen))
    
    ###generate new features
    P_gen2 = P[I_gen[0,:],:]
    ###the first way   
    rand_w = args.dissim * torch.ones([numGen,1])
    P_gen = (torch.mul(P[I_gen[0,:],:], 1 - rand_w) + torch.mul(P[I_gen[1,:],:], rand_w)).numpy()
    #P_gen = np.concatenate((P_gen,P_gen2))
    #print( I_gen[0,:] )
    
    id_gen = I_gen[0,:].squeeze().numpy().astype(int)
    gen_feature_path = os.path.join(args.result_root_path,'gen_features.mat')
    sio.savemat(gen_feature_path, {'features':P_gen, 'ids':id_gen})

def compute_ot(ae_feature_path, alter_target_measure, idxs, args, mode='train'):
    #arguments for training OT
    original_examples = torch.load(ae_feature_path)  # AE encoder latent space features
    print("original_examples=", original_examples.shape)
    target_examples = original_examples.view(original_examples.size(0), -1) 
    #target_examples = F.normalize(target_examples, p=2, dim=1)
    print("target_examples=", target_examples.shape)

    if args.one_attribute:
        blood = target_examples[idxs[0]]
        derma = target_examples[idxs[1]]
        target_examples = torch.cat((blood,derma), dim=0)
        h_P = target_examples
    elif args.two_attribute:
        maglss = target_examples[idxs[0]]
        manonglass = target_examples[idxs[1]]
        femaglass = target_examples[idxs[2]]
        femanonglass = target_examples[idxs[3]]
        target_examples = torch.cat((maglss,manonglass,femaglass,femanonglass), dim=0)
        h_P = target_examples
    else:
        n00 = target_examples[idxs[0]]
        n01 = target_examples[idxs[1]]
        n02 = target_examples[idxs[2]]
        n03 = target_examples[idxs[3]]
        n04 = target_examples[idxs[4]]
        n05 = target_examples[idxs[5]]
        n06 = target_examples[idxs[6]]
        n07 = target_examples[idxs[7]]
        n08 = target_examples[idxs[8]]
        n09 = target_examples[idxs[9]]
        target_examples = torch.cat((n00,n01,n02,n03,n04,n05,n06,n07,n08,n09), dim=0)
        h_P = target_examples

     
    num_P = h_P.shape[0]
    bat_size_P = num_P
    
    dim_y = h_P.shape[1]
    maxIter = args.maxIter
    lr_ot = args.lr_ot
    bat_size_n = args.bat_size_n
    init_num_bat_n = args.init_num_bat_n
    num_gen_x = bat_size_n*init_num_bat_n     #a multiple of bat_size_n
    eps = args.eps
    alter_measture = args.alter_measture 
     
    '''train omt'''
    selected_ot_model_path = os.path.join(args.result_root_path, 'h.pt')
    print(selected_ot_model_path)
    #target_measure_path = os.path.join(args.result_root_path, 'nu.pt')

    if mode=='train':

        p_s = OMT(h_P, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps, alter_measture, alter_target_measure)
        if os.path.exists(selected_ot_model_path): 
            print('The height vector file exists.')
            tuning_omt(p_s, selected_ot_model_path, init_num_bat_n)
        else:
            print('The height vector file does not exist.')
            train_omt(p_s, init_num_bat_n)
                
        torch.save(p_s.d_h, selected_ot_model_path)

    #else:
        #p_s = OMT(h_P, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps, feature_indices)
        #p_s.set_h(torch.load(selected_ot_model_path)) 
       
    if mode=='generate':
        """generate new samples"""
        #torch.manual_seed(1234)
        #feature_indices = torch.randperm(h_P.size(0))
        #h_P = h_P[feature_indices]
        p_s = OMT(h_P, num_P, dim_y, maxIter, lr_ot, bat_size_P, bat_size_n, eps, alter_measture,alter_target_measure,feature_indices)
        p_s.set_h(torch.load(selected_ot_model_path))
        gen_P(p_s, num_gen_x, feature_indices, args) 
      
