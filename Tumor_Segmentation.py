# Code to segment tumor regions from WSI using ResNet18 model

#Load Libraries
import openslide
import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import scipy
import skimage
import scipy.ndimage
import random
import torch
import tables
from sklearn import model_selection
import cv2
from matplotlib.patches import Rectangle
from skimage import measure
import random
import math
import time
import tables
from skimage import io, morphology
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet
import PIL

dataname="Tumor_segmentation"
gpuid=0
edge_weight=1

path = 'Trained model path'
import os
os.chdir(path)


print(torch.cuda.get_device_properties(gpuid))
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(f"./models/{dataname}_best_model_v1.pth")


### Load Model
import torchvision.models as models
model = models.resnet18(pretrained=True).to(device)
model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
model = model.to(device)
model.load_state_dict(checkpoint["model_dict"])
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")


#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::]

# To find white regions in a patch
def find_white_spaces(im_patch):
    gray = cv2.cvtColor(im_patch, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold (blurred,200,255,cv2.THRESH_BINARY)[1]
    return thresh

#Create a bounding box on a WSI
def bbox(binary_im):
    a = np.where(binary_im != 0)
    return np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])


# This function is written to create a sub region of a WSI
def region_selection(io,fname):
    patch_size = 250
    batch_size = 500
    n_white_pixels=[]

    io_shape_orig = io.shape
    arr_out = sklearn.feature_extraction.image._extract_patches(io,(patch_size,patch_size,3),stride_size)
    arr_out_shape = arr_out.shape
    arr_out = arr_out.reshape(-1,patch_size,patch_size,3)
    for batch_arr in divide_batch(arr_out,batch_size):
        aa =[np.sum((find_white_spaces(batch_arr[i,:])>0))/(patch_size*patch_size) for i in range(batch_arr.shape[0])]   
        n_white_pixels = np.append(n_white_pixels,aa,axis=0)
    n_white_mask = n_white_pixels.reshape(arr_out_shape[0],arr_out_shape[1],1,1)
    n_white_mask=np.concatenate(np.concatenate(n_white_mask,1),1)
    n_white_mask = cv2.resize(n_white_mask, (io_shape_orig[1], io_shape_orig[0]), interpolation = cv2.INTER_NEAREST)
    n_white_mask = n_white_mask<0.5

    labels = measure.label(n_white_mask)
    aa = [np.sum(labels == i) for i in np.unique(labels)]
    final_mask = labels == np.argmax(aa[1:])+1
   
    left_x,right_x,top_y,bottom_y =  bbox(final_mask)
    return left_x,right_x,top_y,bottom_y

# Testing WSIs
files=glob.glob('/Path to load WSIs from .. /*.svs')
print(len(files))

# ------ work on files
import sklearn.feature_extraction.image
from sklearn import model_selection
resize = 1
batch_size = 32
patch_size = 250 #should match the value used to train the network
level = 2
stride_size = 1 #Set stride size. This can be [1, patch_size//2]

class_names=["Non-Tumor", "Tumor"]

for filei in files:

    fname=filei
    print(f"working on file: \t {fname}")
                                                
    #########    Image      ###########
    img = openslide.OpenSlide(fname)
    io = np.array(img.read_region((0,0),level,img.level_dimensions[level]).convert('RGB'))
    left_x,right_x,top_y,bottom_y = region_selection(io,fname)# This creates a subsection of a WSI to avoid memory error
    
    
    io = np.array(img.read_region((top_y*2**(level+2), left_x*2**(level+2)),0,((bottom_y-top_y)*2**(level+2), (right_x-left_x)*2**(level+2))).convert('RGB'))
    io = cv2.resize(io, (0, 0), fx=resize, fy=resize)
    io_shape_orig = np.array(io.shape)
    
    
    #add half the stride as padding around the image, so that we can crop it away later
    io = np.pad(io, [(stride_size//2, stride_size//2), (stride_size//2, stride_size//2), (0, 0)], mode="reflect")
    io_shape_wpad = np.array(io.shape)
    
    #pad to match an exact multiple of unet patch size, otherwise last row/column are lost
    npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
    npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

    io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

    arr_out = sklearn.feature_extraction.image._extract_patches(io,(patch_size,patch_size,3),stride_size)
    arr_out_shape = arr_out.shape
    arr_out = arr_out.reshape(-1,patch_size,patch_size,3)
    

    #in case we have a large network, lets cut the list of tiles into batches
    output = np.zeros((0,checkpoint["num_classes"]))
    for batch_arr in divide_batch(arr_out,batch_size):
        
        # This is required to exclude white patches i.e the patches that consist of over 50% of background area
        n_white_pixels =[np.sum((find_white_spaces(batch_arr[i,:])>0))/(patch_size*patch_size) for i in range(batch_arr.shape[0])]
        indxx =np.argwhere(np.array(n_white_pixels)>0.5) #indxx contains the indices of white patches
        
        arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

        # ---- get results
        output_batch = model(arr_out_gpu)
    
         
#         --- pull from GPU and append to rest of output 
        output_batch = output_batch.detach().cpu().numpy()
        for r in range(len(indxx)):
            output_batch[indxx[r][0]]=[1,0] #White patches
        output = np.append(output,output_batch,axis=0)
        
    pred_vals = output


    output = pred_vals.reshape(arr_out_shape[0],arr_out_shape[1],1,1,3)
    output=np.concatenate(np.concatenate(output,1),1)
    output = cv2.resize(output, (io_shape_orig[1], io_shape_orig[0]), interpolation = cv2.INTER_NEAREST)
    
    
    # ****************************************************************************************************
#                                         Plotting
    # ****************************************************************************************************
    
    fig, ax = plt.subplots(1,2, figsize=(14,4))  # 1 row, 2 columns
    ax[0].imshow(io[:io_shape_orig[0],:io_shape_orig[1]])
    ax[0].set_title('Original', fontsize=20)  
        
    ax[1].imshow(np.argmax(output,axis=2))
    ax[1].set_title('Predicted', fontsize=20)   


# ##########################################################
# ######################################################
