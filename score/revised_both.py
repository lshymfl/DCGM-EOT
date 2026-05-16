import numpy as np
import torch
import types
from tqdm import tqdm

from .inception import InceptionV3
#from .fid import calculate_frechet_distance  #, torch_cov, get_statistics
from .score_fid  import get_activations, compute_statistics_of_path, calculate_frechet_distance 

device = torch.device('cuda:0')


def get_inception_and_fid_score(images, data_generate, fid_cache, num_images=None,
                                splits=10, batch_size=50,
                                use_torch=False,
                                verbose=False,
                                parallel=False):
    """when `images` is a python generator, `num_images` should be given"""

    if num_images is None and isinstance(images, types.GeneratorType):
        raise ValueError(
            "when `images` is a python generator, "
            "`num_images` should be given")

    if num_images is None:
        num_images = len(images)

     
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)
        

    if use_torch:
        is_probs = torch.empty((num_images, 1008)).to(device)
    else:
        is_probs = np.empty((num_images, 1008))

    iterator = iter(tqdm(
        images, total=num_images,
        dynamic_ncols=True, leave=False, disable=not verbose,
        desc="get_inception_and_fid_score"))

    start = 0
    while True:
        batch_images = []
        # get a batch of images from iterator
        try:
            for _ in range(batch_size):
                batch_images.append(next(iterator))
        except StopIteration:
            if len(batch_images) == 0:
                break
            pass
        batch_images = np.stack(batch_images, axis=0)
        end = start + len(batch_images)

        # calculate inception feature
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)
            #pred2 = model2(batch_images)
            if use_torch:
                is_probs[start: end] = pred[0]
                print(is_probs)
            else:
                is_probs[start: end] = pred[0].cpu().numpy()
                #print('is_probs=',is_probs.shape)
        start = end

     
    # FID Score
    f = np.load(fid_cache)
    m2, s2 = f['mu'][:], f['sigma'][:]
    #print('m2=',m2)
    #print('s2=',s2)
    f.close()
    
    block_idxFID = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    modelFID = InceptionV3([block_idxFID]).to(device)
    #m1, s1 = calculate_activation_statistics(data_generate, modelFID, batch_size=50, dims=2048,device='cpu')
    #m1, s1 = get_statistics(images, num_images=None, batch_size=50, use_torch=False, verbose=False, parallel=False)
    files = compute_statistics_of_path(data_generate, modelFID, batch_size, dims=2048, device='cuda')
    act = get_activations(files, modelFID, batch_size=50, dims=2048, device='cuda')
    m1 = np.mean(act, axis=0)
    s1 = np.cov(act, rowvar=False)
    #print(m1)
    #print(s1)
    
    '''if use_torch:
        m1 = torch.mean(fid_acts, axis=0)
        print(m1)
        s1 = torch_cov(fid_acts, rowvar=False)
        print(s1)
        m2 = torch.tensor(m2).to(m1.dtype).to(device)
        s2 = torch.tensor(s2).to(s1.dtype).to(device)
    else:
        m1 = np.mean(fid_acts, axis=0)
        print('m1=',m1)
        s1 = np.cov(fid_acts, rowvar=False)
        print('s1=',s1)'''
    fid_score = calculate_frechet_distance(m1, s1, m2, s2, use_torch=use_torch)

    #del is_probs, scores  #,model  #,fid_acts 
    return fid_score
