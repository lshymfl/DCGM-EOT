# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
import numpy as np
import pandas as pd
import os, argparse, time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from score.both import get_inception_and_fid_score
from utils.all_loss import *
from utils.util_hq_attribute import *
from utils.sdot_ot_util import compute_sdot
from utils.otnet_ot_util import compute_otnet
#from utils.ot_util_attribute import compute_ot
from utils.P_loader import P_loader
from sklearn.cluster import KMeans
import random
 
from modelsAE.HQ_new import autoencoder, Encoder, Decoder
from modelsOT.brenier2560 import  DualInputNet, DualInputNet1, DualInputNet2

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_name) 
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs available.")
 
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ae", help="whether to train AE", dest='actions', action='append_const', const='train_ae')
    parser.add_argument("--extract_feature", help="whether to extract latent code with AE encoder", dest='actions', action='append_const', const='extract_feature')
    parser.add_argument("--train_SDOT", help="whether to train (i.e. compute) OT with semi-discrete", dest='actions', action='append_const', const='train_SDOT')
    parser.add_argument("--train_otnet", help="whether to train (i.e. compute) OT with OT-Net", dest='actions', action='append_const', const='train_otnet')
    parser.add_argument("--train_net_ot", help="train ot by network(H,Y,nu)", dest='actions', action='append_const', const='train_net_ot')
    parser.add_argument("--generate_images", help="whether to generate new latent codes", dest='actions', action='append_const', const='generate_images')
    parser.add_argument("--generate_label_images", help="whether to generate new latent codes", dest='actions', action='append_const', const='generate_label_images')
     
    parser.add_argument("--csv_file", help='path to training set directory (for AE)', type=str, metavar="", dest="csv_file", 
    default="/media/user/deeplearning/Advance-LSH/2026-dataset/FFHQ/ffhq_labels.csv")
    parser.add_argument("--data_root_train", help='path to training set directory (for AE)', type=str, metavar="", dest="data_root_train", 
    default="/media/user/deeplearning/Advance-LSH/2026-dataset/FFHQ/trains/") #alltrain
    parser.add_argument("--data_root_test", help='path to testing set directory(for AE)', type=str, metavar="", dest= "data_root_test",
    default="/media/user/deeplearning/Advance-LSH/2026-dataset/FFHQ/test/")  ## 
    parser.add_argument('--testload', default=False, type=str2bool, help='whether test data is needed to validate the model')
    
    #------ parameter of train AE ------
    parser.add_argument('--epochs', type=int, default=100, help='The whole Epochs for AE to train')#260
    parser.add_argument('--batch_size', type=int, default=32, help=' batch size of AE training')
    
    parser.add_argument('--in_channels', type=int, default=3,help='input image number of channels')
    parser.add_argument('--out_ch', type=int, default=3,help='output image number of channels')
    parser.add_argument('--z_channels', type=int, default=16,help='number of features in first layer of AE') ##32/64
    parser.add_argument('--resolution', type=int, default=256,help='original images size') ##32/64
    parser.add_argument('--ch', type=int, default=64 )  
    parser.add_argument('--num_res_blocks', type=int, default=2 )  
    parser.add_argument('--dropout', type=int, default=0.0 )  
    parser.add_argument('--ch_mult',  default=[1,2,4] )  
    parser.add_argument('--attn_resolutions',  default=[]) ## [16,]
    parser.add_argument('--embed_dim', type=int, default=16,help='dimension of feature in hidden space')
    
    parser.add_argument('--lr', type=float, default=2e-5,help='learning rate of AE training')##  5e-4 5e-5
    parser.add_argument('--lpips_weight', default=0.05, type=float, help='if wd=0 and lpips>0, runing mse + lpips')##0.06
    parser.add_argument('--recnums', type=int, default=9,help='vistual numbers of reconstructed images')##
    parser.add_argument('--encoder_param', default=True, type=str2bool, help='Freeze parameter if setup False')
    parser.add_argument('--decoder_param', default=True, type=str2bool, help='Freeze parameter if setup False')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay of optimal Adam')
    parser.add_argument('--opt_batch_size', type=int, default=20, help=' batch size of extract features')
    parser.add_argument('--parallel', default=True, type=str2bool, help='multi gpu training')

     
    #------ parameter of train SDOT ------
    parser.add_argument('--maxItersdot', type=int, default=20,help='max iters of train ot')
    parser.add_argument('--lr_ot', type=int, default=50000e-1,help='learning rate of calculate the update step based on gradient')
    #------ parameter of train OT-Net ------
    parser.add_argument('--maxIterotnet', type=int, default=1,help='max iters of train ot')
    parser.add_argument('--lr_otnet', type=int, default=1e-3,help='learning rate of calculate the update step based on gradient')
    
    parser.add_argument('--bat_size_n', type=int, default=4000,help='Size of mini-batch of Monte-Carlo samples on device')
    parser.add_argument('--init_num_bat_n', type=int, default=40,help='Starting number of mini-batch of Monte-Carlo samples')
    parser.add_argument('--eps', type=int, default=4e-3,help='error of g_norm of train OT')
    parser.add_argument('--N', type=int, default=2,help='cyclic generation of random measures to obtain the height vector and measure')
    parser.add_argument('--randmeasure', default=True, type=str2bool, help='rand measure or equal measure')

    
    #------ parameter of train OT via neural network ------
    parser.add_argument('--netepochs', type=int, default=1000, help='The whole Epochs for AE to train')#260
    parser.add_argument('--netbatch_size', type=int, default=1024, help=' batch size of AE training')## 4096/2e-4/0.9
    parser.add_argument('--feature_dim', type=int, default=65536, help=' dimension of feature')
    parser.add_argument('--measure_dim', type=int, default=1, help=' dimension of measure embedding')
    parser.add_argument('--lr_net', type=float, default=3e-4, help='learning rate of AE training')##  
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay of optimal Adam') 
    parser.add_argument('--train_ratio', type=float, default=0.95, help='training sample ratio') 
    #parser.add_argument('--result_root_path', type=str, default='./FFHQ1/', help='root path to save results')
    #parser.add_argument('--hmodel', type=str, default='./FFHQ1/h_models/Epoch_1996_models.pth', help='root path to h models')
    parser.add_argument('--nums_batch', type=int, default=16,help='The number of batches of h is obtained by pre-training the model')
    parser.add_argument('--train', default=True, type=str2bool, help='train model')
    parser.add_argument('--test_trainsample', default=True, type=str2bool, help='test h via pre-train model')
    parser.add_argument('--alter_measture', default=True, type=str2bool, help='achieving h via alter measure')
    parser.add_argument('--alter_measture_nums', default=70000, type=float, help='achieving h via alter measure')
    parser.add_argument('--cons_samples', default=True, type=str2bool, help='construct training samples')
    parser.add_argument('--one_attribute', default=False, type=str2bool, help='given one attribute measure')
    parser.add_argument('--two_attribute', default=False, type=str2bool, help='given two attribute measure')
    
    #------ generated images -------------
    parser.add_argument('--gen_nums', type=int, default=64,help='max number of generated samples')##
    parser.add_argument('--num_m', type=int, default=1,help='batch_size number of generated samples')##
    parser.add_argument('--gennums', type=int, default=16,help='vistual numbers of generated samples')##
    parser.add_argument('--Lambda', type=int, default=0.1, help='the nearest k samples around current sample')
    parser.add_argument('--label', type=int, default=3, help='Data categories contained in the raw data')
    
    #######################################FID and IS
    parser.add_argument('--result_root_path', type=str, default='./FFHQ_new/', help='root path to save results')
    parser.add_argument("--fid_cache", type=str, metavar="", dest= "fid_cache",default='./stats/Sffhq.train.npz')
    parser.add_argument('--calculate_FID', default=False, type=str2bool, help='mu and sigma of calculated train data')
    
    #######################################dataset
    #parser.add_argument('--dataset', type=str, default='FFHQ', help='reader dataset type')
    #parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--mdb', default=False, type=str2bool, help='train dataset is mdb or images')
 
    
    return parser.parse_args()

if __name__ == "__main__":
    # experiment setting
    args = get_args()
    if args.actions is None:
        actions = ['train_ae', 'extract_feature', 'train_SDOT', 'train_otnet', 'train_net_ot', 'generate_images','generate_label_images']
    else:
        actions = args.actions

    # prepare the training arguments
    RESUME = True #toggles of whether to resume training
        
    subfolders = ['ae_models','rec_imgs','gen_imgs','h_models']
    for i in range(len(subfolders)):
        fpath = os.path.join(args.result_root_path,subfolders[i])
        if not os.path.exists(fpath):
            os.makedirs(fpath)
    ae_model_path = os.path.join(args.result_root_path, 'ae_models')   
    ae_feature_path = os.path.join(args.result_root_path, 'ae_features.pt')  
    h_model_path = os.path.join(args.result_root_path,'h_models') 
    ot_height_path = os.path.join(args.result_root_path, 'h.pt')   
    #Start training and/or generating
    for action in actions:
        if args.mdb:  
            dataset, testset = get_dataset(args) #Get the data and divide it into training sets and test sets
        else:        
            img_transform = transforms.Compose([#transforms.Resize(args.image_size),
                #transforms.RandomHorizontalFlip(),#transforms.RandomVerticalFlip(),#transforms.RandomRotation(5),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([#transforms.Resize(args.image_size),
                transforms.ToTensor(),
            ])
            transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
            #dataset = P_loader(root=args.data_root_train,transform=img_transform)
            dataset = CustomDataset(csv_file=args.csv_file, root_dir=args.data_root_train, transform=img_transform)
            testset = P_loader(root=args.data_root_test,transform=test_transform)
 
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4 )   
        print(len(dataloader.dataset))       
        #########################################################################################################
        model = autoencoder(args.ch, args.out_ch, args.ch_mult, args.num_res_blocks, args.attn_resolutions, args.in_channels, 
            args.resolution, args.z_channels, args.dropout).cuda()
        model_net = DualInputNet(args.feature_dim, args.measure_dim).cuda()
       
        #model.apply(weights_init)
        if args.parallel:
            model = torch.nn.DataParallel(model)  #, device_ids=[0, 1, 2]
   
        if action == 'train_ae':
            train_ae(model,dataloader,testloader,args,resume=RESUME)
        
        if action == 'extract_feature':
            dataloader_stable = DataLoader(dataset, batch_size=args.opt_batch_size, shuffle=False, drop_last=False, num_workers=4)
            extract_feature_ae(model, dataloader_stable, ae_model_path, args)


        ####  alter target measure
        fileName = './FFHQ_new/ffhq_labels.csv'
        data = pd.read_csv(fileName)
        if args.cons_samples:
            target_sample = torch.load(ae_feature_path)
            part = torch.div(torch.ones(args.alter_measture_nums), args.alter_measture_nums) 
            rest = torch.zeros(target_sample.size(0) - part.size(0))
            alter_target_measure = torch.cat((part,rest), dim=0)
            idxs = []
            #print(alter_target_measure)
        elif args.one_attribute:
            column_data = data['Expression']  ## Gender Glass   Expression
            tensor_data = torch.tensor(column_data.values)
            #smile_data = data['Expression']
            #smiling_data = torch.tensor(smile_data.values)
            #race_data = data['beauty']
            #racing_data = torch.tensor(race_data.values)
            greater_than_zero_idx = torch.nonzero(tensor_data == 1).squeeze() 
            less_than_zero_idx = torch.nonzero(tensor_data != 1).squeeze()
            greater_length = greater_than_zero_idx.size(0)
            less_length = less_than_zero_idx.size(0)
            print(greater_length, less_length)
            proportions = torch.tensor([greater_length, less_length])
            weights = 1 / proportions
            normalized_weights = weights / torch.sum(weights)
            print(normalized_weights)  ## gender/Age-normalized_weights//Eyeglasses-0.83/0.17//Expression-0.12/0.88 
            #torch.manual_seed(1234)
            greater = torch.ones(greater_length)
            greater = 0.13*greater/torch.sum(greater)  
            #greater = normalized_weights[0]*greater/torch.sum(greater)  

            less = torch.ones(less_length)
            less = 0.87*less/torch.sum(less)
            #less = normalized_weights[1]*less/torch.sum(less)
            
            alter_target_measure = torch.cat((greater,less), dim=0)
            idxs = [greater_than_zero_idx, less_than_zero_idx]
            print(torch.sum(alter_target_measure))
        elif args.two_attribute:
            gender_data = data['Gender']
            attr_data = data['Glass']
            gender_data = torch.tensor(gender_data.values)
            attr_data = torch.tensor(attr_data.values)
            male_attr_idx = torch.nonzero( torch.logical_and(gender_data > 0, attr_data == 1 ) ).squeeze() 
            female_attr_idx = torch.nonzero( torch.logical_and(gender_data < 0, attr_data == 1 ) ).squeeze()
            male_attr = male_attr_idx.size(0)
            female_attr = female_attr_idx.size(0)
            male_Non_attr_idx = torch.nonzero( torch.logical_and(gender_data > 0, attr_data != 1 ) ).squeeze() 
            female_Non_attr_idx = torch.nonzero( torch.logical_and(gender_data < 0, attr_data != 1 ) ).squeeze()
            male_Non_attr = male_Non_attr_idx.size(0)
            female_Non_attr = female_Non_attr_idx.size(0)
            print('female&attr=',female_attr,'male&attr=',male_attr, 
                 'male&Non-attr=',male_Non_attr,'female&Non-attr=',female_Non_attr)
            proportions = torch.tensor([female_attr,male_attr,male_Non_attr,female_Non_attr])
            weights = 1 / proportions
            normalized_weights = weights / torch.sum(weights)
            print(normalized_weights) 
            #gender&race--h_pred2(0.56,0.35,0.03,0.06)/h_pred3(0.565,0.35,0.03,0.055)/h_pred4(.675,0.245,0.025,0.055)
            #/h_pred5(.68,0.245,0.025,0.05)/h_pred6(.687,0.24,0.025,0.048)
            #gender&glass--(0.635,0.095,0.085,0.185)   #gender&smiling--(0.055,0.08,0.105,0.76)
             #gender&Age--(0.055,0.08,0.105,0.76)
            
            female_attr_measure = torch.ones(female_attr)
            #female_attr_measure = normalized_weights[0]*female_attr_measure/torch.sum(female_attr_measure)
            female_attr_measure = 0.635*female_attr_measure/torch.sum(female_attr_measure)

            male_attr_measure = torch.ones(male_attr)
            #male_attr_measure = normalized_weights[1]*male_attr_measure/torch.sum(male_attr_measure)
            male_attr_measure = 0.095*male_attr_measure/torch.sum(male_attr_measure)

            male_Non_attr_measure = torch.ones(male_Non_attr)
            #male_Non_attr_measure = normalized_weights[2]*male_Non_attr_measure/torch.sum(male_Non_attr_measure)
            male_Non_attr_measure = 0.085*male_Non_attr_measure/torch.sum(male_Non_attr_measure)

            female_Non_attr_measure = torch.ones(female_Non_attr)
            #female_Non_attr_measure = normalized_weights[3]*female_Non_attr_measure/torch.sum(female_Non_attr_measure)
            female_Non_attr_measure = 0.185*female_Non_attr_measure/torch.sum(female_Non_attr_measure)
            
            alter_target_measure = torch.cat((female_attr_measure,male_attr_measure,male_Non_attr_measure,female_Non_attr_measure), dim=0)
            idxs = [female_attr_idx, male_attr_idx,  male_Non_attr_idx,  female_Non_attr_idx]
            print(torch.sum(alter_target_measure))
        else:
            gender_data = data['Gender']
            race_data = data['Race']
            age_data = data['Age']
            gender_data = torch.tensor(gender_data.values)
            race_data = torch.tensor(race_data.values)
            age_data = torch.tensor(age_data.values)
            male_black_young_idx = torch.nonzero( (gender_data >0)& (race_data ==1)&(age_data <45) ).squeeze()  
            male_black_nonyoung_idx = torch.nonzero( (gender_data >0)& (race_data ==1)&(age_data >=45) ).squeeze()
            male_nonblack_young_idx = torch.nonzero( (gender_data >0)& (race_data !=3)&(age_data <45) ).squeeze()  
            male_nonblack_nonyoung_idx = torch.nonzero( (gender_data >0)& (race_data !=3)&(age_data >=45) ).squeeze()  
            male_black_young = male_black_young_idx.size(0)
            male_black_nonyoung = male_black_nonyoung_idx.size(0)
            male_nonblack_young = male_nonblack_young_idx.size(0)
            male_nonblack_nonyoung = male_nonblack_nonyoung_idx.size(0)

            female_black_young_idx = torch.nonzero( (gender_data <0)& (race_data ==1)&(age_data <45) ).squeeze()  
            female_black_nonyoung_idx = torch.nonzero( (gender_data <0)& (race_data ==1)&(age_data >=45) ).squeeze()
            female_nonblack_young_idx = torch.nonzero( (gender_data <0)& (race_data !=3)&(age_data <45) ).squeeze()  
            female_nonblack_nonyoung_idx = torch.nonzero( (gender_data <0)& (race_data !=3)&(age_data >=45) ).squeeze()  
            female_black_young = female_black_young_idx.size(0)
            female_black_nonyoung = female_black_nonyoung_idx.size(0)
            female_nonblack_young = female_nonblack_young_idx.size(0)
            female_nonblack_nonyoung = female_nonblack_nonyoung_idx.size(0)
            print('male_black_young=',male_black_young,'male_black_nonyoung=',male_black_nonyoung,
            'male_nonblack_young=',male_nonblack_young,'male_nonblack_nonyoung=',male_nonblack_nonyoung,
            'female_black_young=',female_black_young,'female_black_nonyoung=',female_black_nonyoung,
            'female_nonblack_young=',female_nonblack_young,'female_nonblack_nonyoung=',female_nonblack_nonyoung)

            proportions = torch.tensor( [male_black_young, male_black_nonyoung, male_nonblack_young,male_nonblack_nonyoung,
                                    female_black_young, female_black_nonyoung, female_nonblack_young,female_nonblack_nonyoung] )
            weights = 1 / proportions
            normalized_weights = weights / torch.sum(weights)
            print(normalized_weights) 

            male_black_young_measure = torch.ones(male_black_young)
            male_B_Y_measure = 0.00002*male_black_young_measure/torch.sum(male_black_young_measure)
            male_black_nonyoung_measure = torch.ones(male_black_nonyoung)
            male_B_NY_measure = 0.43*male_black_nonyoung_measure/torch.sum(male_black_nonyoung_measure)
            male_nonblack_young_measure = torch.ones(male_nonblack_young)
            male_NB_Y_measure = 0.0*male_nonblack_young_measure/torch.sum(male_nonblack_young_measure)
            male_nonblack_nonyoung_measure = torch.ones(male_nonblack_nonyoung)
            male_NB_NY_measure = 0.000031*male_nonblack_nonyoung_measure/torch.sum(male_nonblack_nonyoung_measure)

            female_black_young_measure = torch.ones(female_black_young)
            female_B_Y_measure = 0.00031*female_black_young_measure/torch.sum(female_black_young_measure)
            female_black_nonyoung_measure = torch.ones(female_black_nonyoung)
            female_B_NY_measure = 0.55844008*female_black_nonyoung_measure/torch.sum(female_black_nonyoung_measure)
            female_nonblack_young_measure = torch.ones(female_nonblack_young)
            female_NB_Y_measure = 0.000109*female_nonblack_young_measure/torch.sum(female_nonblack_young_measure)
            female_nonblack_nonyoung_measure = torch.ones(female_nonblack_nonyoung)
            female_NB_NY_measure = 0.01109*female_nonblack_nonyoung_measure/torch.sum(female_nonblack_nonyoung_measure)

            alter_target_measure = torch.cat((male_B_Y_measure,male_B_NY_measure,male_NB_Y_measure,male_NB_NY_measure,
                                female_B_Y_measure,female_B_NY_measure,female_NB_Y_measure,female_NB_NY_measure), dim=0)
            idxs = [male_black_young_idx, male_black_nonyoung_idx, male_nonblack_young_idx, male_nonblack_nonyoung_idx,
            female_black_young_idx, female_black_nonyoung_idx, female_nonblack_young_idx, female_nonblack_nonyoung_idx]
            print(torch.sum(alter_target_measure))
             

        ''' train(compute) OT with OT solver '''
        if action == 'train_SDOT':
            start = time.time()
            compute_sdot(ae_feature_path, alter_target_measure, idxs, args, mode='train')
            end = time.time()
            print("Train_ot done at %.3f seconds." % (end - start))
        
        if action == 'train_otnet':
            start = time.time()
            compute_otnet(ae_feature_path, args, mode='train')
            end = time.time()
            print("Train_ot done at %.3f seconds." % (end - start))

        if action == 'train_net_ot':
            start = time.time()
            train_ot_model(h_model_path, model_net, alter_target_measure,idxs,  args)
            end = time.time()
            print("Train_network_ot done at %.3f seconds." % (end - start))

        if action == 'generate_feature': ##generated new latent feature after OT
            print('Generating features with OT solver...') 
            compute_ot(ae_feature_path, alter_target_measure,idxs,  args, mode='generate')
            torch.cuda.empty_cache()

        ''' decode new latent feature '''    
        if action == 'decode_feature':
            decode_feature(ae_model_path,model,args)
        
        if action == 'generate_image':
            ot_height_path = os.path.join(args.result_root_path, 'h_smile.pt')
            generate_images(model,ae_model_path,ae_feature_path,ot_height_path,idxs,args)
        
         
             