import numpy as np
from pathlib import Path
from spectral_dl.srcnn import SrCNN 
from spectral_dl.dataset import ARPESDataset, NPYArpes
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_ssim import SSIM
import torch
from tqdm import tqdm
import copy
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter


folder_ = "./Cu111_LH19"
epochs = 30
learning_rate = 5e-4
batch_size = 16
flux = 1e3
alpha = 0.7
print_evry = 10

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485],
                     std=[0.229])
])


#dset = ARPESDataset(folder_, flux, tr)
dset = NPYArpes(['./train_data.npy', './train_noise.npy'])
dl = DataLoader(dset, batch_size, shuffle=True,)

val_dset = NPYArpes(['./validation_data.npy', './validation_noise.npy'])
val_dl = DataLoader(dset, batch_size, shuffle=True,)

tb = SummaryWriter()

net = SrCNN().cuda()

loss_mae = nn.L1Loss()
loss_ssim = SSIM()

opt = optim.Adam(net.parameters(), lr=5e-4)
path2weights = './spectral_dl/weights.pt'

best_loss = 1000

#images, labels = next(iter(dl))
#grid = torchvision.utils.make_grid(images.unsqueeze(1))
#tb.add_image("images", grid)
#tb.add_graph(net, images)
#tb.close()

for epoch in range(epochs):
    net.train()
    epoch_loss = 0
    iters = 0
    for (noise, data) in tqdm(dl):
        noise = noise.cuda(non_blocking=True).to(torch.float32)
        #noise = (noise - noise.mean())/noise.std()
        
        data = data.cuda(non_blocking=True).to(torch.float32)
        #data = (data - data.mean())/data.std()
        
        opt.zero_grad()
        out = net(noise.unsqueeze(1))
        loss = (1-alpha)*loss_mae(out, data.unsqueeze(1)) + alpha*loss_ssim(out, data.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        opt.step()
        iters += 1
        epoch_loss += loss.item()
        
    epoch_loss = epoch_loss / iters
        
    if epoch % 50 == 0 and epoch > 0:
        for g in opt.param_groups:
            g['lr'] *= 0.1

    print("epoch: {}, loss: {}".format(epoch, epoch_loss))
    
    tb.add_scalar("Loss", epoch_loss, epoch)

    
    net.eval()
    with torch.no_grad():
        eval_loss = 0
        iters = 0
        for (noise, data) in val_dl:
            noise = noise.cuda(non_blocking=True).to(torch.float32)
            #noise = (noise - noise.mean())/noise.std()
            
            data = data.cuda(non_blocking=True).to(torch.float32)
            out = net(noise.unsqueeze(1))
            loss = (1-alpha)*loss_mae(out, data.unsqueeze(1)) + alpha*loss_ssim(out, data.unsqueeze(1))
            eval_loss += loss.item()
            iters += 1
        
        eval_loss = eval_loss / iters
        
        print("evaluation loss: {}".format(eval_loss))
        
        if eval_loss < best_loss:
            print("found better eval loss")
            best_loss = eval_loss
            torch.save(net.state_dict(), path2weights)
            
    tb.add_scalar("Validation", eval_loss, epoch)


torch.save(net.state_dict(), path2weights)
tb.close()

