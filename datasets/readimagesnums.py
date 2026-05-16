from PIL import Image
import os.path
import glob
import os
''' 
def convertjpg(jpgfile,outdir,width=None,height=None):
  img=Image.open(jpgfile)
  try:
    new_img=img.resize((width,height),Image.BILINEAR)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
  except Exception as e:
    print(e)
 
path = "/home/ubuntu/workspace/data/CelebA_1024/test/mfemale/*.jpg"
jpgfile="/home/ubuntu/workspace/data/CelebA_1024/test/mfemale"
width = 512
height = width
for jpgfile in glob.glob(path):
  convertjpg(jpgfile,"/home/ubuntu/workspace/data/CelebA_512/test/2", width=width, height=height)
'''  
  
 

 

### real images
data_generate = os.path.join("/user79/DiffusionAE-OT/DMAE_bedroom-150/original_imgs/")
imagelist = os.listdir(data_generate)
print(len(imagelist))