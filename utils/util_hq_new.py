import torch
from  torch import nn
from torchvision.utils import save_image
from torch.autograd import Variable
import fnmatch
import os
import scipy.io as sio
from utils.Regularization import Regularization 
from score.revised_both import get_inception_and_fid_score
import numpy as np
import copy
from utils.all_loss import *
import torch.nn.functional as F
from utils.module import *
import lpips
import pandas as pd
#from scipy.stats import shapiro   ######added 

def adjust_learning_rate(learn_rate, optimizer, epoch):
    # Decreasing the learning rate to the factor of 0.1 at epochs 51 and 65
    # with a batch size of 256 this would comply with changing the lr at iterations 12k and 15k
 
    ## cifar10
    if 500 < epoch < 1500:
        lr = learn_rate * 0.1
    elif epoch >= 1500:
        lr = learn_rate * 0.01
    else:
        lr = learn_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_ae(model, trainloader,testloader, args,resume=False):
    ''' train AE model '''
    recnums = args.recnums 
    rows = int(np.sqrt(recnums))

    for test_data in testloader:
        if args.mdb:
            test_img, _ , _ = test_data
        else:
            test_img, _, _ = test_data
            #test_img = DiffusionStep(test_img, args.truncated_T)  ###added
        break
    ###############################add
    for data_data in trainloader:
        if args.mdb:
            data_img,_,_ = data_data
        else:
            data_img, _,_ = data_data
            #data_img = DiffusionStep(data_img1, args.truncated_T)  ###added
        break

    ae_model_path = os.path.join(args.result_root_path,'ae_models')
    if resume:
        for file in os.listdir(ae_model_path):
            print(file)
            model.load_state_dict(torch.load(os.path.join(ae_model_path, file)))
     
    ########################
    criterion = WeightedLoss()
    mse_fn = nn.MSELoss()
    loss_fn = lpips.LPIPS(net='alex').cuda() # 'vgg' 'alex'   'squeeze'
    #criterion = InfoNCELoss(temperature=0.1)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr , betas=(0.5, 0.9))
    #### show model size
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024)) 

    #save input test image
    img_save_path = os.path.join(args.result_root_path,'rec_imgs')
    save_image(data_img[:recnums], os.path.join(img_save_path, 'data_image_input.jpg'),nrow=rows)  ###########################add
    for epoch in range(args.epochs):
        #adjust_learning_rate(args.lr, optimizer, epoch)#######################
        count_train = 0
        loss_train = 0.0
        count_test = 0
        loss_test = 0.0
        lpips_loss = 0.0
        for data in trainloader:   
            if args.mdb:
                img, _, _  = data
            else:
                img,_,_  = data

            img = Variable(img).cuda()            
            # ===================forward=====================
            output,fea = model(img)
            
            mse_loss = mse_fn(output, img)   
            lpips_loss = loss_fn(output, img).mean() 
            loss = (1 - args.lpips_weight)*mse_loss + args.lpips_weight*lpips_loss 
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            count_train += 1
                  
        if args.testload:
            for data in testloader:
                if args.mdb:
                    img, _, _  = data
                else:
                    img, _,_ = data
                img = Variable(img).cuda()
                model.eval()
                with torch.no_grad():
                    output,_ = model(img)
                loss = mse_fn(output, img)
    
                loss_test += loss.item()
                count_test += 1
            loss_test /= count_test
        else:
            loss_test = 0

        loss_train /= count_train
 

        print('epoch [{}/{}], loss_train:{:.8f}, loss_test:{:.8f}'.format(epoch, args.epochs, loss_train, loss_test))#add
        ###################################
        torch.save(model.state_dict(), os.path.join(ae_model_path,'Epoch_{}_sim_autoencoder_{:06f}.pth'.format(epoch, loss_train)))
        #testput,_ = model(test_img.cuda())
        #pic = testput.data.cpu()
        #save_image(pic[:recnums], os.path.join(img_save_path, 'Epoch_{}_test_image_{:06f}_{:06f}.jpg'.format(epoch, loss_train, loss_test)),nrow=rows)
        ###################################add
        dataput,_ = model(data_img.cuda())
        datapic = dataput.data.cpu()
        save_image(datapic[:recnums], os.path.join(img_save_path, 'Epoch_{}_data_image_{:06f}_{:06f}.jpg'.format(epoch, loss_train, loss_test)),nrow=rows)
        

 
def extract_feature_ae(model, dataloader,model_path,args ):  
    ''' extract_feature of AE model '''
    for data in dataloader:
        img, _, _  = data
        with torch.no_grad():
            decoded, z = model(img.cuda())
        break   
    features = torch.empty([len(dataloader.dataset), z.shape[1], z.shape[2], z.shape[3]], dtype=torch.float, requires_grad=False, device='cpu')
    for file in os.listdir(model_path):
        print(file)
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    i = 0
    for data in dataloader:
        if args.mdb:
            img, _, _  = data
        else:
            img, _,_  = data
        with torch.no_grad():
            if args.resolution == 256:
                posterior = model.module.encoder(img.cuda())
            else:
                posterior = model.encoder(img.cuda())
            z = posterior.sample()
        features[i:i+img.shape[0], :, :, :] = z.cpu()   #.squeeze()
        i += img.shape[0]
    #print('Extracted {}/{} features...'.format(i, data_len))
    print('Extracted features complete')
    feature_save_path = os.path.join(args.result_root_path,'ae_features.pt')
    torch.save(features, feature_save_path)
    print(features.shape)

####################################### generate_images ##########################################################
 
def generate_images(model,model_path,feature_path,ot_height_path,args): 

    for file in os.listdir(model_path):
        print(file)   
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    gen_im_path = os.path.join(args.result_root_path,'gen_imgs')
    #os.makedirs(gen_im_path, exist_ok=True)

    feature_z = torch.load(feature_path)
    feature = feature_z.view(feature_z.size(0), -1) 
    h = torch.load(ot_height_path)   
    soft = nn.Softmax(dim=0)
    A = torch.cat([h.unsqueeze(1).cpu(), feature], dim=1).cuda() 
    
    for i in range(args.num_m):
        '''
        z_all = torch.empty([args.gen_nums, feature.shape[1]])        #101 feature, 99 interpolation feature
        for i in range(args.gen_nums):
            t = (i)/ args.gen_nums   # 0,...,100  => [0,1]           
            z_1_2 = (t)*feature[1] + (1-t)*feature[8]
            z_all[i] = z_1_2                
        new_code = z_all 
        ''' 
        Q = torch.randn((args.gen_nums, feature.shape[1]), dtype=torch.float)
        one = torch.ones(args.gen_nums,1) 
        Z = torch.cat([one, Q], dim=1).cuda() 
        I = torch.mm(A, Z.t()) 
        weight = soft(I/args.Lambda).cuda() 
        G = torch.mm(A.t(), weight) 
        new_code = G[1:,].t()
        print(new_code.shape)     
        
        latent = new_code.view(args.gen_nums,feature_z.shape[1],feature_z.shape[2],feature_z.shape[3]).cuda()
        with torch.no_grad():
            if args.resolution == 256:
                y = model.module.decoder(latent)
            else:
                y = model.decoder(latent)
            for k in range(y.size(0)):
                y_gen = y[k,:,:,:]
                save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:08d}_gen.jpg'.format(k+i*args.gen_nums))) 
    
    rows = int(np.sqrt(args.gennums))
    #save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_imgs_{}.jpg'.format(args.Lambda)), nrow=rows, padding=0)
    save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_img_{}.jpg'.format(args.Lambda)), nrow=4, padding=0)

     

 
        
