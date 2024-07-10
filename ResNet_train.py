#Code to train Resnet18 model for Tumor Segmentation

# Load libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import DenseNet
import PIL
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys, glob
import time
import math
import tables
import random
from sklearn.metrics import confusion_matrix


data_path = 'DATA PATH'# Enter path to read the data from
import os
os.chdir(data_path)


#Helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#Specify if we should use a GPU (cuda) or only the CPU
print(torch.cuda.get_device_properties(gpuid))
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')


dataname="data_file"#Enter filename to load the patches from
gpuid=0

#Set parameters
num_classes=2    #number of classes in the data mask that we'll aim to predict
in_channels= 3  #input channel of the data, RGB = 3
batch_size=32
patch_size=250 
drop_rate=0.5 #Set dropout rate
num_epochs = 80
phases = ["train","val"] #Use "train" phase to train the model and "val" phase to evaluate it


torch.cuda.empty_cache()
import torchvision.models as models
model = models.resnet18(pretrained=True).to(device)
    
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
model = model.to(device)

#This defines our dataset class which will be used by the dataloader
class Dataset(object):
    def __init__(self, fname ,img_transform=None):
        #nothing special here, just internalizing the constructor parameters
        self.fname=fname
        self.img_transform=img_transform
        
        with tables.open_file(self.fname,'r') as db:
            self.classsizes=db.root.classsizes[:]
            self.nitems=db.root.imgs.shape[0]
        
        self.imgs = None
        self.labels = None
        
    def __getitem__(self, index):

        with tables.open_file(self.fname,'r') as db:
            self.imgs=db.root.imgs
            self.labels=db.root.labels

            #get the requested image and mask from the pytable
            img = self.imgs[index,:,:,:]
            label = self.labels[index]
        
        
        img_new = img
        
        if self.img_transform is not None:
            img_new = self.img_transform(img)


        return img_new, label, img
    def __len__(self):
        return self.nitems

#Augmentations
img_transform = transforms.Compose([
     transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(size=(patch_size,patch_size),pad_if_needed=True), 
#     transforms.CenterCrop(size=(patch_size,patch_size)), 
#     transforms.RandomResizedCrop(size=patch_size),
    transforms.transforms.Resize(size=patch_size),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.1),
    transforms.ToTensor()
    ])


dataset={}
dataLoader={}
for phase in phases: #now for each of the phases, we're creating the dataloader
                   
    dataset[phase]=Dataset(data_path+'/'+dataname+'_'+phase+'.pytable', img_transform=img_transform)
    dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                shuffle=True, num_workers=0,pin_memory=True) 
    
    print(f"{phase} dataset size:\t{len(dataset[phase])}")


#visualize a single example to verify that it is correct
(img, label, img_old)=dataset["val"][450]
fig, ax = plt.subplots(1,2, figsize=(10,4))  # 1 row, 2 columns

#build output showing patch after augmentation and original patch
ax[0].imshow(np.moveaxis(img.numpy(),0,-1))
ax[1].imshow(img_old)

print(label)

optim = torch.optim.Adam(model.parameters(),lr=.001) #adam optimizer

#we have the ability to weight individual classes, in this case we'll do so based on their presence in the trainingset
#to avoid biasing any particular class
nclasses = dataset["train"].classsizes.shape[1]
class_weight=dataset["train"].classsizes[1,:]
class_weight = torch.from_numpy(1-class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)

print(class_weight) #show final used weights, make sure that they're reasonable before continuing
criterion = nn.CrossEntropyLoss(weight = class_weight)

best_loss_on_val = np.Infinity

start_time = time.time()
for epoch in range(num_epochs):
    all_acc = {key: 0 for key in phases} 
    all_loss = {key: torch.zeros(0).to(device) for key in phases} #keep this on GPU for greatly improved performance
    cmatrix = {key: np.zeros((num_classes,num_classes)) for key in phases}

    for phase in phases: #iterate through both training and validation states
        if phase == 'train':
            model.train()  # Set model to training mode
        else: #when in eval mode, we don't want parameters to be updated
            model.eval()   # Set model to evaluate mode

        for ii , (X, label, img_orig) in enumerate(dataLoader[phase]): #for each of the batches
            X = X.to(device)  # [Nbatch, 3, H, W]
            label = label.type('torch.LongTensor').to(device)  # [Nbatch, 1] with class indices (0, 1, 2,...num_classes)

            with torch.set_grad_enabled(phase == 'train'): 
                prediction = model(X)  # [N, Nclass]
                loss = criterion(prediction, label)
                

                if phase=="train": #in case we're in train mode, need to do back propogation
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss


                all_loss[phase]=torch.cat((all_loss[phase],loss.detach().view(1,-1)))
                
#               #if this phase is a part of validation, compute confusion matrix
                p=prediction.detach().cpu().numpy()
                cpredflat=np.argmax(p,axis=1).flatten()
                yflat=label.cpu().numpy().flatten()
                cmatrix[phase]=cmatrix[phase]+confusion_matrix(yflat,cpredflat, labels=range(nclasses))

        all_acc[phase]=(cmatrix[phase]/cmatrix[phase].sum()).trace()
        all_loss[phase] = all_loss[phase].cpu().numpy().mean()


    print('%s ([%d/%d] %d%%), train loss: %.4f val loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs), 
                                                 epoch+1, num_epochs ,(epoch+1) / num_epochs * 100, all_loss["train"], all_loss["val"]),end="") 

    #if current loss is the best we've seen, save model state with all variables
    if all_loss["val"] < best_loss_on_val:
        best_loss_on_val = all_loss["val"]
        print("  **")
        state = {'epoch': epoch + 1,
         'model_dict': model.state_dict(),
         'optim_dict': optim.state_dict(),
         'in_channels': in_channels,
         'num_classes':num_classes}

        torch.save(state, f"{dataname}_resnet_best_model_v1.pth")

