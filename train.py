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

workers = 2
original_batch_size = 128
image_size = 64
nc = 3  # Number of channels in the training images. For color images this is 3
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
lr = 0.002  # Learning rate for optimizers
beta1 = 0.5  # Beta1 hyper-param for Adam optimizers
ngpu = 1  # Number of GPUs available. Use 0 for CPU mode.
outputPath = "ResultImages/"


def train(epoch, directory, dsPath, dsName):
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    with open(directory + '/vars.json') as f:
        dataJson = json.load(f)

    cudnn.benchmark = True

    # Root directory for dataset
    dataroot = dsPath
    pathNetG = directory + "/gen.pth"
    pathNetD = directory + "/dis.pth"

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=original_batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if dataJson['epochs'] != 0:
        netG.load_state_dict(torch.load(pathNetG))
        print("NetG loaded")

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
        print("NetD loaded")

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(original_batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # Number of training epochs
    num_epochs = epoch

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if dataJson['epochs'] != 0:
        netG.load_state_dict(torch.load(pathNetG))
        print("OptimizerG loaded")
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if dataJson['epochs'] != 0:
        netD.load_state_dict(torch.load(pathNetD))
        print("OptimizerD loaded")

        # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print("Start training")
    netG.train()
    netD.train()

    if not os.path.exists(outputPath + dsName):
        os.makedirs(outputPath + dsName)

    for epoch in range(num_epochs):
        print(f"Epoch {dataJson['epochs'] + 1}")
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
                                  '%s/fake_%s.png' % (outputPath + dsName,
                                                      str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))),
                                  normalize=True)

        dataJson['epochs'] += 1

    netG.eval()
    netD.eval()
    torch.save(netG.state_dict(), pathNetG)
    torch.save(netD.state_dict(), pathNetD)
    with open(directory + "/vars.json", "w") as f:
        json.dump(dataJson, f)
    print("Training finished")


def generate(datasetName, seed, show_images, count_of_images=10, ):
    directory = f"Datasets/{datasetName}"
    random.seed(seed)
    torch.manual_seed(seed)
    with open(directory + "/vars.json") as f:
        dataJson = json.load(f)

    pathNetG = directory + "/gen.pth"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if dataJson['epochs'] != 0:
        netG.load_state_dict(torch.load(pathNetG))
        print("NetG loaded")

    model = RealESRGAN(device, scale=2)
    model.load_weights('RealESRGAN/weights/RealESRGAN_x2.pth')

    def improve_quality(in_model, path):
        image = Image.open(path).convert('RGB')
        sr_image = in_model.predict(image)
        if show_images:
            sr_image.show()
        sr_image.save(path)

    netG.eval()
    with torch.no_grad():
        for i in range(int(count_of_images)):
            z = torch.randn(1, 100, 1, 1, device=device)
            fake = netG(z).detach().cpu()
            dataset_output = outputPath + datasetName
            if not os.path.exists(dataset_output):
                os.makedirs(dataset_output)
            outputPathOneImage = dataset_output + f"/generated_{dataJson['image_iterator']}" + ".png"
            vutils.save_image(fake, outputPathOneImage, normalize=True)
            resize_image(outputPathOneImage, outputPathOneImage, size=(512, 512))
            try:
                print("Trying improve quality of image...")
                improve_quality(model, outputPathOneImage)
                print("Quality improvement finished!")
            except RuntimeError:
                print("Cannot improve quality")
            print(f"Image {dataJson['image_iterator']} saved in {outputPathOneImage}")
            dataJson['image_iterator'] += 1
    with open(directory + '/vars.json', "w") as f:
        json.dump(dataJson, f)

    print("Images generation finished!")


def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print(f"The original image size is {width} wide x {height} high")

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print(f"The resized image size is {width} wide x {height} high")
    resized_image.save(output_image_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


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
