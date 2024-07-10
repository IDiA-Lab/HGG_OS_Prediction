# Code to train ResNet-Cox model in 5 folds cross-validation setting

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
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index

# --- training params
dataname="survival"
gpuid=0
in_channels= 3  #input channel of the data, RGB = 3
batch_size=32
patch_size=250
num_epochs = 100
phases = ["train"] #how many phases did we create databases for?


data_path = 'Path to load the data from'
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


# Specify if we should use a GPU (cuda) or only the CPU
print(torch.cuda.get_device_properties(gpuid))
torch.cuda.set_device(gpuid)
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')


class CoxRegression(nn.Module):
    def __init__(self, nfeat):
        super(CoxRegression, self).__init__()
        self.fc5 = nn.Linear(nfeat, 1)
        self.init_hidden()

    def forward(self, x, coo=None):
        x = self.fc5(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc5.weight)


import torchvision.models as models
class ResnetCox(nn.Module):
    def __init__(self, dropout, pretrained):
        super(ResnetCox, self).__init__()
        self.resnet = models.resnet18(pretrained=True).to(device)
        self.resnet.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        self.resnet.load_state_dict(pretrained["model_dict"]) #Load model weights from the ResNet18 tumor segmentation model
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.dropout = nn.Dropout(dropout)


        for param in self.resnet.parameters():
            param.requires_grad = True 

        self.cox = CoxRegression(512)

    def forward(self, x, coo=None):
        h = self.resnet(x)
        h = self.dropout(h)
        x = self.cox(torch.flatten(h, 1))
        return x


#This defines our dataset class which will be used by the dataloader
class Dataset(object):
    def __init__(self, fname ,img_transform=None):
        #Nothing special here, just internalizing the constructor parameters
        self.fname=fname
        self.img_transform=img_transform
        
        with tables.open_file(self.fname,'r') as db:
            self.nitems=db.root.img.shape[0]
        
        self.img = None
        self.censored = None
        self.survival = None
        self.batch_indx = None
        
        
    def __getitem__(self, index):

        with tables.open_file(self.fname,'r') as db:
            self.img=db.root.img
            self.censored = db.root.OS_censor
            self.survival = db.root.OS
            
            #get the requested image and labels from the pytable
            img = self.img[index,:,:,:]
            censored = self.censored[index]
            survival = self.survival[index]

            
        img_new = img
        
        if self.img_transform is not None:
            img_new = self.img_transform(img)

        return img_new, survival, censored, index, img
    
    def __len__(self):
        return self.nitems
    
    
class SurvivalDataset(Dataset):
    def __init__(self, data):
        self.x = data[0]
        self.y = data[1]
        self.c = data[2]
        self.R = _make_R(np.array(self.y))

        self.indices = list(range(self.x.shape[0]))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.c[idx], self.indices[idx] # indices -> To get R matrix

def _make_R(time):
    assert time.ndim == 1, "expected 1D array"

    # sort in descending order
    o = np.argsort(-time, kind="mergesort")
    n_samples = len(time)
    risk_set = np.zeros((n_samples, n_samples), dtype=np.bool_)
    for i_org, i_sort in enumerate(o):
        ti = time[i_sort]
        k = i_org
        while k < n_samples and ti == time[o[k]]:
            k += 1
        risk_set[i_sort, o[:k]] = True
    return risk_set



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
for phase in phases:                     
    dataset[phase]=Dataset(data_path+'/'+dataname+'_'+phase+'.pytable', img_transform=img_transform)
    dataLoader[phase]=DataLoader(dataset[phase], batch_size=batch_size, 
                                shuffle=True, num_workers=0,pin_memory=True) 
    print(f"{phase} dataset size:\t{len(dataset[phase])}")

#visualize a single example to verify that it is correct
(img, survival, censored, batch_indx, img_old)=dataset["train"][40]
fig, ax = plt.subplots(1,2, figsize=(10,4))  # 1 row, 2 columns

#build output showing patch after augmentation and original patch
ax[0].imshow(np.moveaxis(img.numpy(),0,-1))
ax[1].imshow(img_old)

# print(survival)
print(survival, censored, batch_indx)


from lifelines.utils import concordance_index as cindex
from sklearn.metrics import explained_variance_score,r2_score,mean_squared_error,mean_absolute_error

class PartialNLL(nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    def forward(self, theta, R, censored):
        theta = theta.double()
        censored = censored.double()
        exp_theta = torch.exp(theta)
        observed = 1 - censored
        num_observed = torch.sum(observed)
        loss = -(torch.sum((theta.reshape(-1)- torch.log(torch.sum((exp_theta), 0))) * observed) / num_observed)


        if np.isnan(loss.data.tolist()):
            for a,b in zip(theta, exp_theta):
                print(a,b)

        return loss
    
def get_evaluations(recon_x, x):
    pred, true = recon_x.cpu().data.numpy(), x.cpu().data.numpy()
    evs = explained_variance_score(pred, true)
    r2 = r2_score(pred, true)
    mse = mean_squared_error(pred, true)
    mae = mean_absolute_error(pred, true)
    float_list = [evs, r2, mse, mae] 
    return ["%.3f"%item for item in float_list]

def get_cindex(y, y_pred, c):
    try:
        return cindex(y, y_pred, c)
    except Exception as e:
        print(e)
        print(y)
        print(y_pred)
        print(c)
        return 0.0
    
def coxph_loss(event, riskset, predictions):
    pred_t = tf.transpose(predictions)
    rr = logsumexp_masked(pred_t, riskset, axis=1, keepdims=True)

    losses = tf.multiply(event, rr - predictions)
    loss = tf.reduce_mean(losses)
    return loss


from lifelines.utils import concordance_index
def model_train(epoch, fold, train_loader):
    model.train()
    total_loss = 0.0
    total = 0.0
    ci = []
    
    for ii , (data) in enumerate(train_loader): #for each of the batches
        X, survival, censored, batch_indx, img_orig = data
        train_dataset = SurvivalDataset(data)

        batch_R = train_dataset.R
        X = X.to(device)  # [Nbatch, 3, H, W]
        optim.zero_grad()
        prediction = model(X) 
        loss = loss_function(prediction, batch_R, censored.to(device)).cpu()
        loss.backward()
        optim.step()

        total_loss += loss.data.tolist()
        total += len(survival)
        
        # Compute Concordance index
        event_observed = np.array([1 if v==0 else 0 for v in censored])
        survival = survival.cpu()
        prediction = -prediction.reshape(-1).data.cpu()

        ci.append(concordance_index(survival, prediction, event_observed))

#     Mean Concordance index across all batches
    print('====> Epoch: {} Fold {} Total train loss: {:.4f} c-index: {:.4f}'.format(epoch, fold, total_loss, np.mean(ci)))
    return {"train loss":total_loss, "c-index":np.mean(ci)}




def model_valid(epoch, fold, val_loader):
    model.eval()

    total_loss = 0.0
    total = 0.0
    ci = []
    for ii , (data) in enumerate(val_loader): #for each of the batches
#         print(ii)
        X, survival, censored, batch_indx, img_orig = data
        valid_dataset = SurvivalDataset(data)

        batch_R = valid_dataset.R
        X = X.to(device)  # [Nbatch, 3, H, W]
        prediction = model(X) 
        loss = loss_function(prediction, batch_R, censored.to(device)).cpu()
        total_loss += loss.data.tolist()
        total += len(survival)
        
        # Compute Concordance index
        event_observed = np.array([1 if v==0 else 0 for v in censored])
        survival = survival.cpu()
        prediction = -prediction.reshape(-1).data.cpu()

        ci.append(concordance_index(survival, prediction, event_observed))

#     Mean Concordance index across all batches
    print('====> Epoch: {} Fold {} Total valid loss: {:.4f} c-index: {:.4f}'.format(epoch, fold, total_loss, np.mean(ci)))
    return {"valid loss":total_loss, "c-index":np.mean(ci)}


dataname="Give name to save the model weights"
pretrained = torch.load(f"./models/{dataname}_resnet_best_model.pth")
model = ResnetCox(0.1,pretrained).to(device)
model.train()
print(model)
loss_function = PartialNLL()
total_loss = 0
total = 0
ci = []
bs = 32
epochs = 100


optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

best_loss = np.Infinity
best_CI = 0


#5 Folds Cross-Validation    
kfold = KFold(n_splits = 5, shuffle = True)
for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset["train"])):
    print("Fold"+str(fold))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(dataset['train'], batch_size = 32, sampler = train_sampler)
    val_loader = DataLoader(dataset['train'], batch_size = 32, sampler = val_sampler)
    
   
    for epoch in range(1, epochs + 1):
        training = model_train(epoch, fold, train_loader)
        validation = model_valid(epoch, fold, val_loader)
           
        # SAVE BEST MODEL using best loss on the validation data
        if validation["valid loss"] < best_loss:
            best_loss = validation["valid loss"]
      
            print('====> Saving model with validation loss: {:.4f} **'.format(validation["valid loss"]))        
            state = {'model_dict': model.state_dict(),
                 'best_loss_on_val': validation["valid loss"]}
        
            model_path = "./models/ResNetCox_model_fold"+str(fold)+".pth"
            torch.save(state, model_path)

