
from __future__ import print_function
import argparse
import os
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import *
from network import *
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet | mnist')
parser.add_argument('--dataroot', default='./datasets/', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--is_continue', type=int, default=1, help='Use pre-trained model')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=256, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=55, help='number of epochs to train for')
parser.add_argument('--mu', type=float, default=1.0, help='weight of Cycle cWonsistency')
parser.add_argument('--W', type=float, default=1.0, help='Wake')
parser.add_argument('--N', type=float, default=1.0, help='NREM')
parser.add_argument('--R', type=float, default=1.0, help='REM')
parser.add_argument('--epsilon', type=float, default=0.0, help='amount of noise in wake latent space')
parser.add_argument('--nf', type=int, default=64, help='filters factor')
parser.add_argument('--drop', type=float, default=0.0, help='probably of drop out')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lmbd', type=float, default=0.5, help='convex combination factor for REM')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='dd', help='folder to output images and model checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='The ID of the specified GPU')
parser.add_argument('--nowake', type=bool, default=False, help='No Wake')
parser.add_argument('--nonrem', type=bool, default=False, help='No NREM')
parser.add_argument('--norem', type=bool, default=False, help='No REM')

opt, unknown = parser.parse_known_args()
print(opt)

# specify the gpu id if using only 1 gpu
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

# where to save samples and training curves
dir_files = './results/'+opt.dataset+'/'+opt.outf
# where to save model
dir_checkpoint = './checkpoints/'+opt.dataset+'/'+opt.outf

try:
    os.makedirs(dir_files)
except OSError:
    pass
try:
    os.makedirs(dir_checkpoint)
except OSError:
    pass

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset, unorm, img_channels = get_dataset(opt.dataset, opt.dataroot, opt.imageSize)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)



# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
batch_size = opt.batchSize

# setup networks
netG = Generator(ngpu, nz=nz, ngf=opt.nf, img_channels=img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu, nz=nz, ndf=opt.nf, img_channels=img_channels,  p_drop=opt.drop)
netD.apply(weights_init)
# send to GPU
netD.to(device)
netG.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []

def load_weights(path=dir_checkpoint+'/trained.pth'):
    print("Loading weights from", path)
    if os.path.exists(path):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        netG.load_state_dict(checkpoint['generator'])
        netD.load_state_dict(checkpoint['discriminator'])
        optimizerG.load_state_dict(checkpoint['g_optim'])
        optimizerD.load_state_dict(checkpoint['d_optim'])
        d_losses = checkpoint.get('d_losses', [float('inf')])
        g_losses = checkpoint.get('g_losses', [float('inf')])
        r_losses_real = checkpoint.get('r_losses_real', [float('inf')])
        r_losses_fake = checkpoint.get('r_losses_fake', [float('inf')])
        kl_losses = checkpoint.get('kl_losses', [float('inf')])
        print('Start training from loaded model...')
    else:
        print('No pre-trained model detected, please do training...')
        exit()


# loss functions
dis_criterion = nn.BCELoss() # discriminator
rec_criterion = nn.MSELoss() # reconstruction

# tensor placeholders
dis_label = torch.zeros(opt.batchSize, dtype=torch.float32, device=device)
real_label_value = 1.0
fake_label_value = 0

eval_noise = torch.randn(batch_size, nz, device=device)

# this program is a pysimpleGUI live demo

import PySimpleGUI as sg
import cv2

def process_image(file_path):
    real_image = cv2.imread(file_path)
    # convert to tensor
    real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
    real_image = cv2.resize(real_image, (opt.imageSize, opt.imageSize))
    real_image = real_image.transpose((2, 0, 1))
    real_image = real_image.reshape(3, opt.imageSize, opt.imageSize)
    real_image = torch.from_numpy(real_image)
    real_image = real_image.type(torch.FloatTensor)
    real_image = real_image.to(device)
    real_image = real_image.unsqueeze(0)

    print(real_image.shape)
    # we need [64, 3, 32, 32]

    # generate the image
    latent_output, dis_output = netD(real_image)
    latent_output_noise = latent_output + opt.epsilon*torch.randn(batch_size, nz, device=device) # noise transformation
    reconstructed_image = netG(latent_output_noise, reverse=False)

    reconstructed_image = unorm(reconstructed_image)
    vutils.save_image(reconstructed_image[0].data, 'temp.png', nrow=1)

    img = cv2.imread('temp.png')
    img = cv2.resize(img, (400, 400))
    cv2.imwrite('temp.png', img)

    return "temp.png"


'''
two images, one is the original image, the other is the generated image
textbox path and import button to import the image from the local file
generate button to generate the image
'''

# PySimpleGUI GUI
layout = [
    [sg.Text("Select weights file:")],
    [sg.Input(key="weights_path", enable_events=True
              ), sg.FileBrowse()],
    [sg.Text("Select an image file:")],
    [sg.Input(key="file_path"), sg.FileBrowse()],
    [sg.Button("Process"), sg.Button("Exit")],
    [sg.Image(key="-IMAGE-", size=(400, 400))],
]

window = sg.Window("Image Processing GUI", layout, finalize=True)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "Process":
        file_path = values["file_path"]
        if file_path:
            new_path = process_image(file_path)

            # Display the processed image in the GUI
            window["-IMAGE-"].update(
                filename=new_path,
                size=(400, 400),
            )
    elif event == "weights_path":
        weights_path = values["weights_path"]
        if weights_path:
            load_weights(weights_path)


window.close()