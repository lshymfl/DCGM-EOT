import os
import shutil
import random

 
data_dir = '/user79/dataset/ImageNet/train/1/'

subfold_dir = '/user79/dataset/ImageNet/training/'
 
num_subfolders = 900
#samples_per_subfolder = 100 

 
for i in range(num_subfolders):
    subfolder_path = os.path.join(subfold_dir, f'{i}')
    os.makedirs(subfolder_path, exist_ok=True)

file_list = os.listdir(data_dir)


samples_per_folder = len(file_list) // num_subfolders


for i in range(num_subfolders):
    subfolder_path = os.path.join(subfold_dir, f'{i}')


    start_idx = i * samples_per_folder
    end_idx = (i + 1) * samples_per_folder if i < num_subfolders - 1 else len(file_list)

    samples = file_list[start_idx:end_idx]

    for sample in samples:
        src_path = os.path.join(data_dir, sample)
        dst_path = os.path.join(subfolder_path, sample)
        shutil.move(src_path, dst_path)

print('end')