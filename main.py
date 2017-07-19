from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import config as cf

import os
import sys
import time
import argparse
import datetime
import random

from torch.autograd import Variable
from data_loader import ImageFolder
from networks import *

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | cifar10')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the width & height of the input image')

parser.add_argument('--nz', type=int, default=100, help='size of latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')

parser.add_argument('--nEpochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=2e-4')
parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam optimizer, default=0.5')

parser.add_argument('--nGPU', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--outf', default='./checkpoints/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

use_cuda = torch.cuda.is_available()
cudnn.benchmark = True
use_cuda = torch.cuda.is_available()

######################### Data Preperation
print("\n[Phase 1] : Data Preperation")
print("| Preparing %s dataset..." %(opt.dataset))

dset_transforms = transforms.Compose([
    transforms.Scale(opt.imageSize),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[opt.dataset], cf.std[opt.dataset])
])

if (opt.dataset == 'cifar10'):
    dataset = dset.CIFAR10(
        root='/home/bumsoo/Data/GAN/cifar10/',
        download=True,
        transform=dset_transforms
    )
elif (opt.dataset == 'mnist'):
    dataset = dset.MNIST(
        root='/home/bumsoo/Data/GAN/mnist/',
        download=True,
        transform=dset_transforms
    )
elif (opt.dataset == 'cell') :
    dataset = ImageFolder(
        root='/home/bumsoo/Data/GAN/cell/',
        transform=dset_transforms
    )
else:
    print("Error | Dataset must be one of mnist | cifar10")
    sys.exit(1)

print("| Consisting data loader for %s..." %(opt.dataset))
loader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = opt.batchSize,
    shuffle = True
)

######################### Model Setup
print("\n[Phase 2] : Model Setup")
ndf = opt.ndf
ngf = opt.ngf

if(opt.dataset == 'cifar10'):
    nc = 3
elif(opt.dataset == 'cell'):
    nc = 3
elif(opt.dataset == 'mnist'):
    nc = 1
else:
    print("Error : Dataset must be one of \'mnist | cifar10 | cell\'")
    sys.exit(1)

print("| Consisting Discriminator with ndf=%d" %ndf)
print("| Consisting Generator with z=%d" %opt.nz)
netD = Discriminator(ndf, nc)
netG = Generator(opt.nz, ngf, nc)

if(use_cuda):
    netD.cuda()
    netG.cuda()
    cudnn.benchmark = True

######################### Loss & Optimizer
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))

######################### Global Variables
noise = torch.FloatTensor(opt.batchSize, opt.nz, 1, 1)
real = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)
real_label, fake_label = 1, 0

noise = Variable(noise)
real = Variable(real)
label = Variable(label)

if(use_cuda):
    noise = noise.cuda()
    real = real.cuda()
    label = label.cuda()

######################### Training Stage
print("\n[Phase 4] : Train model")
for epoch in range(1, opt.nEpochs+1):
    for i, (images) in enumerate(loader): # We don't need the class label information

        ######################### fDx : Gradient of Discriminator
        netD.zero_grad()

        # train with real data
        real.data.resize_(images.size()).copy_(images)
        label.data.resize_(images.size(0)).fill_(real_label)

        output = netD(real) # Forward propagation, this should result in '1'
        errD_real = 0.5 * torch.mean((output-label)**2) # criterion(output, label)
        errD_real.backward()

        # train with fake data
        label.data.fill_(fake_label)
        noise.data.resize_(images.size(0), opt.nz, 1, 1)
        noise.data.normal_(0, 1)

        fake = netG(noise) # Create fake image
        output = netD(fake.detach()) # Forward propagation for fake, this should result in '0'
        errD_fake = 0.5 * torch.mean((output-label)**2) # criterion(output, label)
        errD_fake.backward()
        #### Appendix ####
        #### var.detach() = Variable(var.data), difference in computing trigger

        errD = errD_fake + errD_real
        optimizerD.step()

        ######################### fGx : Gradient of Generator
        netG.zero_grad()
        label.data.fill_(real_label)
        output = netD(fake) # Forward propagation of generated image, this should result in '1'
        errG = 0.5 * torch.mean((output - label)**2) # criterion(output, label)
        errG.backward()
        optimizerG.step()

        ######################### LOG
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d] Loss(D): %.4f Loss(G): %.4f '
                %(epoch, opt.nEpochs, i, len(loader), errD.data[0], errG.data[0]))
        sys.stdout.flush()

    ######################### Visualize
    if(i%1 == 0):
        print(": Saving current results...")
        vutils.save_image(
            fake.data,
            '%s/fake_samples_%03d.png' %(opt.outf, epoch),
            normalize=True
        )

######################### Save model
torch.save(netG.state_dict(), '%s/netG.pth' %(opt.outf))
torch.save(netD.state_dict(), '%s/netD.pth' %(opt.outf))
