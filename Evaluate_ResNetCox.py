#Code to evaluate ResNet-Cox model

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


# --- Set params
dataname="survival"
gpuid=0
patch_size=250
phases = ["test"] 

data_path = 'Path to load the data from'
import os
os.chdir(data_path)

#helper function for pretty printing of current time and remaining time
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

#specify if we should use a GPU (cuda) or only the CPU
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
    def __init__(self):
        super(ResnetCox, self).__init__()

        self.resnet = models.resnet18(pretrained=True).to(device)
        self.resnet.fc = nn.Linear(in_features=512, out_features=2, bias=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.cox = CoxRegression(512)

    def forward(self, x, coo=None):
        h = self.resnet(x)
        x = self.cox(torch.flatten(h, 1))
        return x

model = ResnetCox().to(device)
checkpoint = torch.load(f"./models/ResNetCox_model.pth")
model.load_state_dict(checkpoint["model_dict"])#Load pretrained model weights

print(model)


#This defines our dataset class which will be used by the dataloader
class Dataset(object):
    def __init__(self, fname ,img_transform=None):
        #nothing special here, just internalizing the constructor parameters
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
            
            #get the requested image and mask from the pytable
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
for phase in phases: #now for each of the phases, we're creating the dataloader
                     #interestingly, given the batch size, i've not seen any improvements from using a num_workers>0
    
    dataset[phase]=Dataset(data_path+'/'+dataname+'_'+phase+'.pytable', img_transform=img_transform)
    dataLoader[phase]=DataLoader(dataset[phase], batch_size=32, 
                                shuffle=True, num_workers=0,pin_memory=True) 
    print(f"{phase} dataset size:\t{len(dataset[phase])}")

#visualize a single example to verify that it is correct
(img, survival, censored, batch_indx, img_old)=dataset["test"][40]
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
def model_test(input_data):
    model.eval()

    total_loss = 0.0
    total = 0.0
    ci = []
    for ii , (data) in enumerate(input_data): #for each of the batches
        X, survival, censored, batch_indx, img_orig = data
        test_dataset = SurvivalDataset(data)

        batch_R = test_dataset.R
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
    print('Total test loss: {:.4f} c-index: {:.4f}'.format(total_loss, np.mean(ci)))
    return {"Test loss":total_loss, "c-index":np.mean(ci)}


from lifelines.utils import concordance_index

loss_function = PartialNLL()
Test = model_test(dataLoader['test'])    



