import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from audio_utils import *
from audio_network import *
from inception import *

parser = argparse.ArgumentParser()
parser.add_argument('--imsize',
                    type=int,
                    default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--nz',
                    type=int,
                    default=256,
                    help='size of the latent z vector')
parser.add_argument('--epochs',
                    type=int,
                    default=50,
                    help='number of epochs to train for')
parser.add_argument('--epsilon',
                    type=float,
                    default=0.0,
                    help='amount of noise in wake latent space')
parser.add_argument('--lrG',
                    type=float,
                    default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--lrD',
                    type=float,
                    default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--ngpu',
                    type=int,
                    default=4,
                    help='number of GPUs to use')
parser.add_argument('--outf',
                    default='audio_default',
                    help='folder to output images and model checkpoints')
parser.add_argument('--nonrem', type=bool, default=False, help='No NREM')
parser.add_argument('--norem', type=bool, default=False, help='No REM')

args = parser.parse_args()
args.dataset = 'spectrogram'
args.dataroot = './sc09/'
args.W = 1.0
args.N = 1.0
args.R = 1.0
args.nf = 64
args.drop = 0.0
args.beta1 = 0.5
args.lmbd = 0.5

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dir_files = './audio_results/' + args.dataset + '/' + args.outf
dir_checkpoint = './audio_checkpoints/' + args.dataset + '/' + args.outf

for dirs in [dir_files, dir_checkpoint]:
    os.makedirs(dirs, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset, unorm, img_channels = get_dataset(args.dataset, args.dataroot,
                                           args.imsize)
ngpu = int(args.ngpu)
nz = int(args.nz)
batch_size = 1
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=2,
                                         drop_last=True)


# setup networks
netG = Generator(ngpu, nz=nz, ngf=args.nf, img_channels=img_channels)
netG.apply(weights_init)
netD = Discriminator(ngpu,
                     nz=nz,
                     ndf=args.nf,
                     img_channels=img_channels,
                     p_drop=args.drop)
netD.apply(weights_init)

netD.to(device)
netG.to(device)

# optimizers: Adam
optimizerD = optim.Adam(netD.parameters(),
                        lr=args.lrD,
                        betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(),
                        lr=args.lrG,
                        betas=(args.beta1, 0.999))
d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []


def load_weights(path=dir_checkpoint + '/trained.pth'):
    global d_losses, g_losses, r_losses_real, r_losses_fake, kl_losses
    if os.path.exists(path):
        print('Loading from previous checkpoint...')
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
        print(f'Training from checkpoint (Epoch {len(d_losses)})...')
    else:
        print('No checkpoint found, training from scratch...')


load_weights()

# loss functions
dis_criterion = nn.BCELoss()  # discriminator
rec_criterion = nn.MSELoss()  # reconstruction

# initialize the variables
dis_label = torch.zeros(batch_size, dtype=torch.float32, device=device)
REAL_VALUE = 1.0
FAKE_VALUE = 0.0

eval_noise = torch.randn(batch_size, nz, device=device)

for epoch in range(len(d_losses), args.epochs):

    save_loss_D = []
    save_loss_G = []
    save_loss_R_real = []
    save_loss_R_fake = []
    save_norm = []
    save_kl = []

    for i, data in enumerate(dataloader, 0):

        ############################
        # Wake (W)
        ###########################
        # Discrimination wake
        optimizerD.zero_grad()
        optimizerG.zero_grad()

        print(data.shape)
        print(data)
        real_image, label = data
        real_image, label = real_image.to(device), label.to(device)
        latent_output, dis_output = netD(real_image)
        latent_output_noise = latent_output + args.epsilon * torch.randn(
            batch_size, nz, device=device)  # noise transformation
        dis_label[:] = REAL_VALUE  # should be classified as real
        dis_errD_real = dis_criterion(dis_output, dis_label)
        if args.R > 0.0:  # if GAN learning occurs
            (dis_errD_real).backward(retain_graph=True)

        # KL divergence regularization
        kl = kl_loss(latent_output)
        kl.backward(retain_graph=True)

        # reconstruction Real data space
        reconstructed_image = netG(latent_output_noise, reverse=False)
        rec_real = rec_criterion(reconstructed_image, real_image)
        if args.W > 0.0:
            (args.W * rec_real).backward()
        optimizerD.step()
        optimizerG.step()
        # compute the mean of the discriminator output (between 0 and 1)
        D_x = dis_output.cpu().mean()
        latent_norm = torch.mean(torch.norm(latent_output.squeeze(),
                                            dim=1)).item()

        ###########################
        # NREM perturbed dreaming (N)
        ##########################
        rec_fake = torch.zeros(1, device=device)
        if not args.nonrem:
            optimizerD.zero_grad()
            latent_z = latent_output.detach()

            with torch.no_grad():
                nrem_image = netG(latent_z)
                occlusion = Occlude(drop_rate=random.random(),
                                    tile_size=random.randint(1, 8))
                occluded_nrem_image = occlusion(nrem_image, d=1)
            latent_recons_dream, _ = netD(occluded_nrem_image)
            rec_fake = rec_criterion(latent_recons_dream,
                                     latent_output.detach())
            if args.N > 0.0:
                (args.N * rec_fake).backward()
            optimizerD.step()

        ###########################
        # REM adversarial dreaming (R)
        ##########################
        dis_errG = torch.zeros(1, device=device)
        dis_errD_fake = torch.zeros(1, device=device)
        dis_output = torch.zeros(batch_size, 1, device=device)

        if not args.norem:
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            lmbd = args.lmbd
            noise = torch.randn(batch_size, nz, device=device)
            if i == 0:
                latent_z = 0.5 * latent_output.detach() + 0.5 * noise
            else:
                latent_z = 0.25 * latent_output.detach(
                ) + 0.25 * old_latent_output + 0.5 * noise

            dreamed_image_adv = netG(
                latent_z, reverse=True)  # activate plasticity switch
            latent_recons_dream, dis_output = netD(dreamed_image_adv)
            dis_label[:] = FAKE_VALUE  # should be classified as fake
            dis_errD_fake = dis_criterion(dis_output, dis_label)
            if args.R > 0.0:  # if GAN learning occurs
                dis_errD_fake.backward(retain_graph=True)
                optimizerD.step()
                optimizerG.step()
            dis_errG = -dis_errD_fake

        D_G_z1 = dis_output.cpu().mean()

        old_latent_output = latent_output.detach()

        ###########################
        # Compute average losses
        ###########################
        save_loss_G.append(dis_errG.item())
        save_loss_D.append((dis_errD_fake + dis_errD_real).item())
        save_loss_R_real.append(rec_real.item())
        save_loss_R_fake.append(rec_fake.item())
        save_norm.append(latent_norm)
        save_kl.append(kl.item())

        if i % 100 == 0:
            loss_d = np.mean(save_loss_D)
            loss_g = np.mean(save_loss_G)
            loss_r_real = np.mean(save_loss_R_real)
            loss_r_fake = np.mean(save_loss_R_fake)
            latent_norm_mean = np.mean(save_norm)
            print(
                f"EPOCH {epoch} | BATCH {i} | Loss_D: {loss_d:.4f} | Loss_G: {loss_g:.4f} | Loss_R_real: {loss_r_real:.4f} | Loss_R_fake: {loss_r_fake:.4f} | D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f} | latent_norm : {latent_norm_mean:.4f} | KL : {np.mean(save_kl):.4f}"
            )
            compare_img_rec = torch.zeros(batch_size * 2, real_image.size(1),
                                          real_image.size(2),
                                          real_image.size(3))
            with torch.no_grad():
                reconstructed_image = netG(latent_output)
            compare_img_rec[::2] = real_image
            compare_img_rec[1::2] = reconstructed_image
            vutils.save_image(unorm(compare_img_rec[:128]),
                              f'{dir_files}/recon_{epoch:03d}.png',
                              nrow=8)
            fake = unorm(dreamed_image_adv)
            vutils.save_image(fake[:64].data,
                              f'{dir_files}/fake_{epoch:03d}.png',
                              nrow=8)

    d_losses.append(np.mean(save_loss_D))
    g_losses.append(np.mean(save_loss_G))
    r_losses_real.append(np.mean(save_loss_R_real))
    r_losses_fake.append(np.mean(save_loss_R_fake))
    kl_losses.append(np.mean(save_kl))
    save_fig_losses(epoch, d_losses, g_losses, r_losses_real, r_losses_fake,
                    kl_losses, None, None, dir_files)

    # save checkpoint
    torch.save(
        {
            'generator': netG.state_dict(),
            'discriminator': netD.state_dict(),
            'g_optim': optimizerG.state_dict(),
            'd_optim': optimizerD.state_dict(),
            'd_losses': d_losses,
            'g_losses': g_losses,
            'r_losses_real': r_losses_real,
            'r_losses_fake': r_losses_fake,
            'kl_losses': kl_losses,
        }, dir_checkpoint + '/trained.pth')

    # for testing
    if epoch == 1:
        torch.save(
            {
                'generator': netG.state_dict(),
                'discriminator': netD.state_dict(),
            }, dir_checkpoint + '/trained2.pth')

    print("Checkpoint saved")
