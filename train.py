# import argparse
import json
import os
import random
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from RealESRGAN.realesrgan import RealESRGAN


def main(epoch, directory, dsPath):
    # Set random seed for reproducibility manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    with open(directory + '/vars.json') as f:
        dataJson = json.load(f)

    cudnn.benchmark = True

    # Root directory for dataset
    dataroot = dsPath
    pathNetG = directory + "/gen.pth"
    pathNetD = directory + "/dis.pth"
    outputPath = "ResultImages"

    # Number of workers for dataloader
    workers = 2

    # Batch size during training
    batch_size = 128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Learning rate for optimizers
    lr = 0.002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    def generate(count_of_images=10):
        model = RealESRGAN(device, scale=4)
        model.load_weights('RealESRGAN/weights/RealESRGAN_x4.pth')

        netG.eval()
        with torch.no_grad():
            for i in range(count_of_images):
                z = torch.randn(1, 100, 1, 1, device=device)
                fake = netG(z).detach().cpu()
                outputPathOneImage = outputPath + f"/generated_{dataJson['image_iterator']}" + ".png"
                vutils.save_image(fake, outputPathOneImage, normalize=True)
                resize_image(outputPathOneImage, outputPathOneImage, size=(512, 512))
                improve_quality(model, outputPathOneImage)
                dataJson['image_iterator'] += 1
        with open("vars.json", "w") as f:
            json.dump(dataJson, f)

        print("Images have been generated!")

    def improve_quality(model, path):
        image = Image.open(path).convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(path)

    def resize_image(input_image_path, output_image_path, size):
        original_image = Image.open(input_image_path)
        width, height = original_image.size
        print('The original image size is {wide} wide x {height} '
              'high'.format(wide=width, height=height))

        resized_image = original_image.resize(size)
        width, height = resized_image.size
        print('The resized image size is {wide} wide x {height} '
              'high'.format(wide=width, height=height))
        resized_image.show()
        resized_image.save(output_image_path)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    # Generator Code

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)
            return output

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if dataJson['epochs'] != 0:
        netG.load_state_dict(torch.load(pathNetG))
        print("loaded netG")
    print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            if input.is_cuda and self.ngpu > 1:
                output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            else:
                output = self.main(input)

            return output.view(-1, 1).squeeze(1)

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if dataJson['epochs'] != 0:
        netD.load_state_dict(torch.load(pathNetD))
        print("loaded netD")
    print(netD)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Number of training epochs
    num_epochs = epoch

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if dataJson['epochs'] != 0:
        netG.load_state_dict(torch.load(pathNetG))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
        print("loaded netG and optimizerG")
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if dataJson['epochs'] != 0:
        netD.load_state_dict(torch.load(pathNetD))
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        print("loaded netD and optimizerD")

    print("Start training")
    netG.train()
    netD.train()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 30 == 0:
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_%s.png' % (outputPath, str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))),
                                  normalize=True)

        dataJson['epochs'] += 1
        # do checkpointing
    netG.eval()
    netD.eval()
    torch.save(netG.state_dict(), pathNetG)
    # torch.save(netG.state_dict(), dataroot + str(datetime.now().strftime("%d-%m-%Y_%H:%M:%S")) + "_gen.pth")
    torch.save(netD.state_dict(), pathNetD)
    # torch.save(netD.state_dict(), dataroot + str(datetime.now().strftime("%d-%m-%Y_%H:%M:%S")) + "_dis.pth")
    with open("vars.json", "w") as f:
        json.dump(dataJson, f)
    print("Finish training")
    generate(1)
    input()


if __name__ == '__main__':
    main()
