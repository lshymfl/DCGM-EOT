import os
import shutil

 
parent_dir = '/user79/dataset/bedroom/training/'

 
target_dir = '/user79/dataset/bedroom/train/1/'

 
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)

   
    if not os.path.isdir(folder_path):
        continue

 
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
 
        shutil.move(image_path, os.path.join(target_dir, image_file))
        #shutil.copy(image_path, os.path.join(target_dir, image_file))
