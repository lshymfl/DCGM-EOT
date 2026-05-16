import torch
from  torch import nn
from torchvision.utils import save_image
from torch.autograd import Variable
import fnmatch
import os
import scipy.io as sio
import torch.nn.functional as F
from utils.Regularization import Regularization 
from score.revised_both import get_inception_and_fid_score
import numpy as np
import copy
from utils.all_loss import *
import torch.nn.functional as F
from utils.module import *
import lpips
import pandas as pd
import clip
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
            test_img, _ = test_data
        else:
            test_img, _, _ = test_data
            #test_img = DiffusionStep(test_img, args.truncated_T)  ###added
        break
    ###############################add
    for data_data in trainloader:
        if args.mdb:
            data_img, _ = data_data
        else:
            data_img, _ = data_data
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
                img, _ = data
            else:
                img,_ = data

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
                    img, _ = data
                else:
                    img, _, _ = data
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
        img, _ = data
        with torch.no_grad():
            decoded, z = model(img.cuda())
        break   
    features = torch.empty([len(dataloader.dataset), z.shape[1], z.shape[2], z.shape[3]], dtype=torch.float, requires_grad=False, device='cpu')
    labels = torch.empty([len(dataloader.dataset)])
    for file in os.listdir(model_path):
        print(file)
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    i = 0
    for data in dataloader:
        if args.mdb:
            img, _ = data
        else:
            img, label = data
        with torch.no_grad():
            if args.resolution == 256:
                posterior = model.module.encoder(img.cuda())
            else:
                posterior = model.encoder(img.cuda())
            z = posterior.sample()
        features[i:i+img.shape[0], :, :, :] = z.cpu()   #.squeeze()
        labels[i:i+img.shape[0]] = label.cpu()   #.squeeze()
        i += img.shape[0]

    print('Extracted features complete')
    feature_save_path = os.path.join(args.result_root_path,'ae_features.pt')
    torch.save(features, feature_save_path)
    print("features=", features.shape)
    #labels_save_path = os.path.join(args.result_root_path,'labels.pt')
    #torch.save(features, labels_save_path)
    #print("labels=", labels.shape)

####################################### generate_images ##########################################################
def generate_images(model,model_path,feature_path,ot_height_path,args): 

    for file in os.listdir(model_path):
        print(file)   
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
 
    feature_z = torch.load(feature_path)
    feature = feature_z.view(feature_z.size(0), -1) 
    h = torch.load(ot_height_path)   
    soft = nn.Softmax(dim=0)
    A = torch.cat([h.unsqueeze(1).cpu(), feature], dim=1)  #.cuda() 
    
    for i in range(args.num_m):

        Q = torch.randn((args.gen_nums, feature.shape[1]), dtype=torch.float)
        one = torch.ones(args.gen_nums,1) 
        Z = torch.cat([one, Q], dim=1)   #.cuda() 
        I = torch.mm(A, Z.t()) 
        weight = soft(I/args.Lambda)  #.cuda() 
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
                save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:08d}_gen.png'.format(k+i*args.gen_nums))) 
    
    rows = int(np.sqrt(args.gennums))
    save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_imgs_{}.png'.format(args.Lambda)), nrow=rows)

    if args.calculate_FID:
        #(IS, IS_std), FID = get_inception_and_fid_score(y.cpu().numpy(), args.fid_cache, num_images=args.gen_nums, use_torch=False, verbose=True)
        FID = get_inception_and_fid_score(y.cpu().numpy(), gen_im_path, args.fid_cache, num_images=args.gen_nums,
                                splits=10, batch_size=50,
                                use_torch=False,
                                verbose=True,
                                parallel=False)
        print(FID)
    
####################################### generate_label_images ##########################################################
def generate_label_images(model,model_path,feature_path,ot_height_path,args): 

    for file in os.listdir(model_path):
        print(file)   
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    gen_im_path = os.path.join(args.result_root_path,'gen_imgs')

    data = pd.read_csv(args.csv_file)
    column_data = data['label']
    labels = torch.tensor(column_data.values)
    num_classes = int(torch.max(labels) + 1)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    generate_label = torch.tensor([args.label])
    one_hot_generate_label = F.one_hot(generate_label, num_classes=num_classes)

    hadamard_product = one_hot_generate_label * one_hot_labels
    non_zero_rows = hadamard_product.abs().sum(dim=1) != 0
    non_zero_idx = torch.nonzero(non_zero_rows).squeeze() 

    feature_z = torch.load(feature_path)
    feature = feature_z.view(feature_z.size(0), -1)
    h = torch.load(ot_height_path) 

    feature_new = feature[non_zero_idx]
    h_new = h[non_zero_idx]

    soft = nn.Softmax(dim=0)
    A = torch.cat([h_new.unsqueeze(1).cpu(), feature_new], dim=1)
    for i in range(args.num_m):
        Q = torch.randn((args.gen_nums, feature.shape[1]), dtype=torch.float)
        one = torch.ones(args.gen_nums,1) 
        Z = torch.cat([one, Q], dim=1)   #.cuda() 
        I = torch.mm(A, Z.t()) 
        weight = soft(I/args.Lambda)  #.cuda() 
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
                save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:08d}_gen.png'.format(k+i*args.gen_nums))) 
    
    rows = int(np.sqrt(args.gennums))
    save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_imgs_{}.png'.format(args.Lambda)), nrow=rows)

    if args.calculate_FID:
        #(IS, IS_std), FID = get_inception_and_fid_score(y.cpu().numpy(), args.fid_cache, num_images=args.gen_nums, use_torch=False, verbose=True)
        FID = get_inception_and_fid_score(y.cpu().numpy(), gen_im_path, args.fid_cache, num_images=args.gen_nums,
                                splits=10, batch_size=50,
                                use_torch=False,
                                verbose=True,
                                parallel=False)
        print(FID)

####################################### generate_text_images ##########################################################
def generate_text_images(model,model_path,feature_path,ot_height_path,args):
    
    for file in os.listdir(model_path):
        print(file)   
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    gen_im_path = os.path.join(args.result_root_path,'gen_imgs')

    data = pd.read_csv(args.csv_file)
    column_data = data['label']
    text_labels = column_data.tolist()     #fox_idx = torch.nonzero(labels == 0).squeeze() 
    ####  text label is convert to text code by CLIP in original data ###
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(text_labels).to(device)
    ###  choose text as label to generate images  ###
    gen_text = clip.tokenize([args.text]).to(device)
    with torch.no_grad():
        text_features = model_clip.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        generate_text_label = model_clip.encode_text(gen_text)
        generate_text_label /= generate_text_label.norm(dim=-1, keepdim=True)

    hadamard_product = generate_text_label * text_features
    row_sums = torch.sum(hadamard_product, dim=1)
    max_index = torch.nonzero(row_sums == torch.max(row_sums) ).squeeze() 

    feature_z = torch.load(feature_path)
    feature = feature_z.view(feature_z.size(0), -1)
    h = torch.load(ot_height_path) 

    feature_new = feature[max_index.cpu()]
    h_new = h[max_index.cpu()]

    soft = nn.Softmax(dim=0)
    A = torch.cat([h_new.unsqueeze(1).cpu(), feature_new], dim=1)

    for i in range(args.num_m):
        Q = torch.randn((args.gen_nums, feature.shape[1]), dtype=torch.float)
        one = torch.ones(args.gen_nums,1) 
        Z = torch.cat([one, Q], dim=1)   #.cuda() 
        I = torch.mm(A, Z.t()) 
        weight = soft(I/args.Lambda)  #.cuda() 
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
                save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:08d}_gen.png'.format(k+i*args.gen_nums))) 
    
    rows = int(np.sqrt(args.gennums))
    save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_imgs_{}_{}.png'.format(args.Lambda,args.text)), nrow=rows)  