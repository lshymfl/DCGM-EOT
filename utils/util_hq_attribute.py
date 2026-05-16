import torch
from  torch import nn
from torchvision.utils import save_image
from torch.autograd import Variable
import fnmatch
import os, time
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
from tqdm import trange, tqdm 
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
            test_img, _, _ = test_data
        else:
            test_img, _, _ = test_data
            #test_img = DiffusionStep(test_img, args.truncated_T)  ###added
        break
    ###############################add
    for data_data in trainloader:
        if args.mdb:
            data_img,_ = data_data
        else:
            data_img, _  = data_data
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
                img,_   = data
                #print(img.shape)

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
                    img, _  = data
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
    for file in os.listdir(model_path):
        print(file)
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    i = 0
    for data in dataloader:
        if args.mdb:
            img, _, _  = data
        else:
            img, _   = data
        with torch.no_grad():
            if args.resolution == args.resolution:
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

#####################################################################################################################
def train_ot_model(h_model_path, model_net, alter_target_measure, idxs, args):
    if args.train:
        target_sample_path = os.path.join(args.result_root_path,'Y_sample.pt')
        target_measure_path = os.path.join(args.result_root_path,'nu_sample.pt')
        height_vector_path = os.path.join(args.result_root_path,'h_sample.pt')

        target_sample = torch.load(target_sample_path)#.cuda()   ###  cpu
        target_measure = torch.load(target_measure_path)#.cuda()
        height_vector = torch.load(height_vector_path)#.cuda()

        num_samples = target_sample.size(0)
        print(target_sample.shape)
        num_train_samples = int(args.train_ratio * num_samples)
        num_test_samples = num_samples - num_train_samples

        #torch.manual_seed(123)
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]

        target_sample_train, target_measure_train, height_vector_train = target_sample[train_indices].detach(), target_measure[train_indices].detach(), height_vector[train_indices].detach()
        target_sample_test, target_measure_test, height_vector_test = target_sample[test_indices], target_measure[test_indices], height_vector[test_indices]

        model_net.train()
        mse_fn = nn.MSELoss()   
        optimizer = torch.optim.Adam(model_net.parameters(), lr=args.lr_net, weight_decay=args.weight)#SGD/Adam/RMSprop
        #optimizer, scheduler = init_optimizer(model_net, args.lr, args.netepochs)

        # save loss data of train models
        train_losses = []
        test_losses = []
        start = time.time()
        for epoch in range(args.netepochs):

            count_train = 0
            loss_train = 0.0
            index = torch.randperm(target_sample_train.size(0))
            target_sample_train = target_sample_train[index]
            target_measure_train = target_measure_train[index]
            height_vector_train = height_vector_train[index]
 
            for i in trange(0, int(target_sample_train.size(0)/args.netbatch_size), 1, desc='train model'):
                batch_sample = target_sample_train[i*args.netbatch_size:(i+1)*args.netbatch_size]
                batch_measure = target_measure_train[i*args.netbatch_size:(i+1)*args.netbatch_size]
                batch_h = height_vector_train[i*args.netbatch_size:(i+1)*args.netbatch_size]

                optimizer.zero_grad()
                #print(batch_sample.shape)
                #print(batch_measure.shape)
                #batch_measure = batch_measure.cpu()
                #print(batch_measure)
                out_h = model_net(batch_sample.cuda(), batch_measure.unsqueeze(1)).squeeze()
                loss = mse_fn(out_h, batch_h)
                #print(loss)
                #loss = loss + args.weight * torch.norm(out_h, 2)
               
                loss.backward()
                optimizer.step()
                 
                #scheduler.step()

                loss_train += loss.item()
                count_train += 1
            
            loss_train /= count_train
            train_losses.append(loss_train)

            ## test models
            with torch.no_grad():
                #print(target_sample_test.shape)
                #print(target_measure_test.shape)
                out_h_test = model_net(target_sample_test.cuda(), target_measure_test.unsqueeze(1)).squeeze()
                #out_h_test = model_net(target_sample_test, args.netbatch_size) 
            test_loss = mse_fn(out_h_test, height_vector_test)
            test_loss = test_loss.detach() 
            test_losses.append(test_loss) 
            #test_loss = 0
             

            print('epoch [{}/{}], loss_train:{:.6f}, loss_test:{:.6f}'.format(epoch, args.netepochs, loss_train, test_loss) )
            if (epoch+1)%10==0:
                torch.save(model_net.state_dict(), os.path.join(h_model_path,'Epoch_{}_models.pth'.format(epoch, loss_train)) )
        
        end = time.time()
        print("Train_ot done at %.3f seconds." % (end - start)) 
        train_loss_data = np.array(train_losses)
        test_loss_data = np.array([x.detach().cpu().item() if torch.is_tensor(x) else x for x in test_losses])
        #print('test_loss_data=',test_loss_data, type(test_loss_data))
        data_dict = {'arr1': train_loss_data, 'arr2': test_loss_data}
        mat_path = os.path.join(args.result_root_path,'train_test_loss.mat')
        sio.savemat(mat_path, data_dict)

    else:
        if args.test_trainsample:
            target_sample_path = os.path.join(args.result_root_path,'Y_sample.pt')
            target_measure_path = os.path.join(args.result_root_path,'nu_sample.pt')
            height_vector_path = os.path.join(args.result_root_path,'h_sample.pt')
            target_sample = torch.load(target_sample_path).cuda() 
            target_measure = torch.load(target_measure_path).cuda()
            height_vector = torch.load(height_vector_path).cuda()
        else:
            target_sample_path = os.path.join(args.result_root_path,'ae_features.pt')
            tag_samples = torch.load(target_sample_path)
            tag_samples = tag_samples.view(tag_samples.size(0), -1) 
            '''if args.one_attribute:
                gr = tag_samples[idxs[0]]
                lr = tag_samples[idxs[1]]
                alter_tag_samples = torch.cat((gr,lr), dim=0)
            elif args.two_attribute:
                maglss = tag_samples[idxs[0]]
                manonglass = tag_samples[idxs[1]]
                femaglass = tag_samples[idxs[2]]
                femanonglass = tag_samples[idxs[3]]
                alter_tag_samples = torch.cat((maglss,manonglass,femaglass,femanonglass), dim=0)
            else:
                MBY = tag_samples[idxs[0]]
                MBNY = tag_samples[idxs[1]]
                MNBY = tag_samples[idxs[2]]
                MNBNY = tag_samples[idxs[3]]
                FMBY = tag_samples[idxs[4]]
                FMBNY = tag_samples[idxs[5]]
                FMNBY = tag_samples[idxs[6]]
                FMNBNY = tag_samples[idxs[7]]
                alter_tag_samples = torch.cat((MBY,MBNY,MNBY,MNBNY,FMBY,FMBNY,FMNBY,FMNBNY), dim=0)
            target_sample = alter_tag_samples.cuda()'''
            target_sample = tag_samples.cuda()


            if args.alter_measture:
                target_measure = alter_target_measure.cuda()
            else:
                torch.manual_seed(123)
                total = torch.randint(0, target_sample.size(0), (target_sample.size(0),))
                target_measure = torch.div( total, sum(total) ).cuda()
                #target_measure = torch.div(torch.ones(target_sample.size(0)), target_sample.size(0)).cuda()
                '''class_num1 = [2130, 21456, 105, 1517, 6830, 44625, 403, 21332, 364, 26126, 1536, 59808]
                class_num = sorted(class_num1, reverse=True)
                total_num = sum(class_num)  
    
                class_num = torch.tensor(class_num, dtype=torch.float64)
                class_ratio = class_num / total_num  
                
                target_measure = []
                for i in range(class_num.shape[0]):
                    measure = torch.ones(int(class_num[i]))*class_ratio[i]
                    target_measure.append(measure)  
                target_measure = torch.cat(target_measure)/torch.sum(torch.cat(target_measure))  
                target_measure = target_measure.cuda()'''
                
            print(target_measure)
            

        model_net.load_state_dict(torch.load(args.hmodel))
        h = torch.empty([target_sample.size(0)])
        lens = int(target_sample.size(0)/args.nums_batch)
        for i in range(args.nums_batch):
            target_sample0 = target_sample[i*lens:(i+1)*lens]
            target_measure0 = target_measure[i*lens:(i+1)*lens]
            h_batch = model_net(target_sample0, target_measure0.unsqueeze(1))
            #print(h_batch)
            h[i*lens:(i+1)*lens] = Variable(h_batch.squeeze().cpu() ,requires_grad=False)
        #h = Variable(h,requires_grad=False)
        torch.save(h, os.path.join(args.result_root_path, 'h_pred.pt') )
        '''
        height_vector_path = os.path.join(args.result_root_path,'h_ori.pt')
        height_vector = torch.load(height_vector_path).cuda()
        ori_h = height_vector.detach().cpu().numpy()
        print('h_original=',ori_h.shape, ori_h)
        predh = h.detach().cpu().numpy()
        print('h_predicate=',predh.shape, predh)

        data_dict = {'arr1': ori_h, 'arr2': predh}
        mat_path = os.path.join(args.result_root_path,'trueH_predH.mat')
        sio.savemat(mat_path, data_dict)
        '''

####################################### generate_images ##########################################################

def generate_images(model,model_path,feature_path,ot_height_path,args): 

    for file in os.listdir(model_path):
        print(file)   
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    gen_im_path = os.path.join(args.result_root_path,'gen_imgs')
    #os.makedirs(gen_im_path, exist_ok=True)

    feature_z = torch.load(feature_path)
    feature = feature_z.view(feature_z.size(0), -1) 
    print("feature=", feature, feature.shape)
    h = torch.load(ot_height_path)
    #h = (h - torch.min(h)) / (torch.max(h) - torch.min(h))
    print("h=", h, h.shape)   
    soft = nn.Softmax(dim=0)
    A = torch.cat([h.unsqueeze(1).cpu(), feature], dim=1).cuda() 
    
    for i in range(args.num_m):
         
         
        #torch.manual_seed(i)
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
            if args.resolution == args.resolution:
                y = model.module.decoder(latent)
            else:
                y = model.decoder(latent)
            for k in range(y.size(0)):
                y_gen = y[k,:,:,:]
                save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:08d}_gen.jpg'.format(k+i*args.gen_nums))) 
    
    rows = int(np.sqrt(args.gennums))
    save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_imgs_{}.jpg'.format(args.Lambda)), nrow=rows, padding=0)
    #save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_img_{}.jpg'.format(args.Lambda)), nrow=4, padding=0)
    
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
    Q = torch.randn((args.gen_nums, feature_new.shape[1]), dtype=torch.float)
    one = torch.ones(args.gen_nums,1)
    Z = torch.cat([one, Q], dim=1)
    I = torch.mm(A, Z.t())
    weight = soft(I/args.Lambda)
    G = torch.mm(A.t(), weight)
    new_code = G[1:,].t()
    print(new_code.shape)      
    
    latent = new_code.view(args.gen_nums,feature_z.shape[1],feature_z.shape[2],feature_z.shape[3]).cuda()
    rows = int(np.sqrt(args.gennums))
    gen_im_path = os.path.join(args.result_root_path,'gen_imgs')
    k = int(args.gen_nums/args.num_m)
    images_label = []#####
    for i in range(k):
        z = latent[i*args.num_m:(i+1)*args.num_m,:,:,:]
        with torch.no_grad():
            if args.resolution == 256:
                y = model.module.decoder(z)
            else:
                y = model.decoder(z)
        print(y.shape)
        images_label.append(y)#####
        for k in range(args.num_m):
            y_gen = y[k,:,:,:]
            save_image(y_gen.cpu(), os.path.join(gen_im_path, 'img_{0:07d}_gen.png'.format(k+i*args.num_m))) 
    save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_imgs_{}.jpg'.format(args.Lambda)), nrow=rows)
    #save_image(y[0:args.gennums].cpu(), os.path.join(args.result_root_path, 'gen_50_{}.jpg'.format(args.Lambda)), nrow=10, padding=0)


    if args.calculate_FID:
        #(IS, IS_std), FID = get_inception_and_fid_score(y.cpu().numpy(), args.fid_cache, num_images=args.gen_nums, use_torch=False, verbose=True)
        FID = get_inception_and_fid_score(y.cpu().numpy(), gen_im_path, args.fid_cache, num_images=args.gen_nums,
                                splits=10, batch_size=50,
                                use_torch=False,
                                verbose=True,
                                parallel=False)
        print(FID)


####################################### decode_feature ##########################################################
def decode_feature(model,model_path,args): 
    ''' decode new latent feature '''
    for file in os.listdir(model_path):
        print(file)   
        model.load_state_dict(torch.load(os.path.join(model_path, file)))
    gen_im_pair_path = os.path.join(args.result_root_path,'gen_imgs_sdot')
    
    gen_feature_path = os.path.join(args.result_root_path,'gen_features.mat')
    feature_dict = sio.loadmat(gen_feature_path)
    features = feature_dict['features']
    ids = feature_dict['ids']
    print(features.shape)
            
    num_feature = features.shape[0]
    z = torch.from_numpy(features).cuda()
    z = z.view(num_feature,16,16,16)
    
    num_m = 250        
    k = int(args.max_gen_samples/num_m)
    print(args.max_gen_samples)
    for i in range(k):
        z1 = z[i*num_m:(i+1)*num_m,:,:,:]
        with torch.no_grad():
            y1 = model.module.decoder(z1)
        print(y1.shape)
        for k in range(num_m):
            y_gen = y1[k,:,:,:]
            save_image(y_gen.cpu(), os.path.join(gen_im_pair_path, 'img_{0:06d}_gen.jpg'.format(k+i*num_m)))
            
    if args.calculate_FID:
        #(IS, IS_std), FID = get_inception_and_fid_score(y.cpu().numpy(), args.fid_cache, num_images=args.gen_nums, use_torch=False, verbose=True)
        FID = get_inception_and_fid_score(y1.cpu().numpy(), gen_im_pair_path, args.fid_cache, num_images=args.gen_nums,
                                splits=10, batch_size=50,
                                use_torch=False,
                                verbose=True,
                                parallel=False)
        print(FID)
    

     

 
        
