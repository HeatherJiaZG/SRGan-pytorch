import argparse
import os
from math import log10

import pytorch_ssim
from data_utils import load_training_data, load_val_data, do_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=90, type=int, help='training images crop size')
parser.add_argument('--upscale', default=4, type=int, choices=[2, 4, 8],
                    help='upscale factor')
parser.add_argument('--epochs', default=100, type=int, help='train epoch number')


if __name__ == '__main__':
    opt = parser.parse_args()
    
    crop_size = opt.crop_size
    upscale = opt.upscale_factor
    epoch = opt.epochs
    
    train_set = load_training_data('DIV2K_train_HR', crop_size=crop_size, upscale=upscale)
    val_set = load_val_data('DIV2K_valid_HR', upscale=upscale)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    
    netG = Generator(upscale)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    generator_loss = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_loss.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    
    for epoch in range(1, epoch + 1):
        trainning_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        for data, target in train_loader:
            g_update_first = True
            batch_size = data.size(0)
            trainning_results['batch_sizes'] += batch_size
    
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            generated_img = netG(z)
    
            netD.zero_grad()
            real_out = netD(real_img).mean()
            generate_out = netD(generated_img).mean()
            d_loss = 1 - real_out + generate_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()
    

            netG.zero_grad()
            g_loss = generator_loss(generate_out, generated_img, real_img)
            g_loss.backward()
            
            generated_img = netG(z)
            generate_out = netD(generated_img).mean()
            
            
            optimizerG.step()
            trainning_results['g_loss'] += g_loss.item() * batch_size
            trainning_results['d_loss'] += d_loss.item() * batch_size
            trainning_results['d_score'] += real_out.item() * batch_size
            trainning_results['g_score'] += generate_out.item() * batch_size
    
        netG.eval()
        out_path = 'training_results/SRF_' + str(upscale) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [do_transform()(val_hr_restore.squeeze(0)), do_transform()(hr.data.cpu().squeeze(0)),
                     do_transform()(sr.data.cpu().squeeze(0))])
                     
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    

        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (upscale, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (upscale, epoch))

        results['d_loss'].append(trainning_results['d_loss'] / trainning_results['batch_sizes'])
        results['g_loss'].append(trainning_results['g_loss'] / trainning_results['batch_sizes'])
        results['d_score'].append(trainning_results['d_score'] / trainning_results['batch_sizes'])
        results['g_score'].append(trainning_results['g_score'] / trainning_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(upscale) + '_train_results.csv', index_label='Epoch')
