import argparse
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
import cv2

from RealESRGAN.realesrgan import RealESRGAN

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--mode', type=int, required=True, default=1, help='0 - create dataset, 1 - train mode, 2 - generate images, 3 - create video')
parser.add_argument('--imageCount', type=int, default=1, help='count of images to create')
parser.add_argument('--datasetName', required=True, help='the name for a new dataset')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)
path_to_dataset = opt.outf + '/' + opt.datasetName
print("Folder path to output images and model checkpoints: " + path_to_dataset)

if opt.manualSeed is None:
    manualSeed = random.randint(1, 10000)
else:
    manualSeed = int(opt.manualSeed)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if int(opt.mode) == 1 or int(opt.mode) == 2:
    with open(path_to_dataset + "/vars.json") as f:
        dataJson = json.load(f)

    cudnn.benchmark = True

    # Root directory for dataset
    if opt.dataroot is None:
        dataroot = dataJson["directory"]
    else:
        dataroot = opt.dataroot
    
    print("Current path to dataset: " + dataroot)
    pathNetG = path_to_dataset + "/gen.pth"
    pathNetD = path_to_dataset + "/dis.pth"
    outputPath = path_to_dataset + "/ResultImages"

    # Number of workers for dataloader
    workers = int(opt.workers)

    # Batch size during training
    train_batch_size = int(opt.batchSize)

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = int(opt.imageSize)

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = int(opt.nz)

    # Size of feature maps in generator
    ngf = int(opt.ngf)

    # Size of feature maps in discriminator
    ndf = int(opt.ndf)

    # Learning rate for optimizers
    lr = float(opt.lr)

    # Beta1 hyperparam for Adam optimizers
    beta1 = float(opt.beta1)

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = int(opt.ngpu)

    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    assert dataset
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)

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

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(train_batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

if int(opt.mode) == 0:
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    try:
        os.makedirs(path_to_dataset)
    except OSError:
        pass

    try:
        os.makedirs(path_to_dataset + '/ResultImages')
    except OSError:
        pass

    with open(path_to_dataset + "/vars.json", "w") as f:
        f.write('{"directory": "' + opt.dataroot + '", "image_iterator": 0, "epochs": 0}')
    print("Finish creating dataset")
elif int(opt.mode) == 1:
    # Number of training epochs
    num_epochs = int(opt.epochs)

    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    if dataJson['epochs'] != 0:
        netG.load_state_dict(torch.load(pathNetG))
        print("loaded netG and optimizerG")
    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if dataJson['epochs'] != 0:
        netD.load_state_dict(torch.load(pathNetD))
        print("loaded netD and optimizerD")
    
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    
    print("Start training")
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
                                  '%s/fake_%s.png' % (outputPath, str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))),
                                  normalize=True)
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % outputPath,
                                  normalize=True)

        dataJson['epochs'] += 1
    
    # do checkpointing
    torch.save(netG.state_dict(), pathNetG)
    # torch.save(netG.state_dict(), dataroot + str(datetime.now().strftime("%d-%m-%Y_%H:%M:%S")) + "_gen.pth")
    torch.save(netD.state_dict(), pathNetD)
    # torch.save(netD.state_dict(), dataroot + str(datetime.now().strftime("%d-%m-%Y_%H:%M:%S")) + "_dis.pth")
    with open(path_to_dataset + "/vars.json", "w") as f:
        json.dump(dataJson, f)
    print("Finish training")
elif int(opt.mode) == 2:
    seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)


    def generate(count_of_images=10):
        model = RealESRGAN(device, scale=4)
        model.load_weights('RealESRGAN/weights/RealESRGAN_x4.pth')

        netG = Generator(ngpu).to(device)
        netG.apply(weights_init)
        if dataJson['epochs'] != 0:
            netG.load_state_dict(torch.load(pathNetG))
            #optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
            print("loaded netG and optimizerG")

        with torch.no_grad():
            for i in range(count_of_images):
                z = torch.randn(1, 100, 1, 1, device=device)
                fake = netG(z).detach().cpu()
                outputPathOneImage = outputPath + f"/generated_{dataJson['image_iterator']}" + ".png"
                vutils.save_image(fake, outputPathOneImage, normalize=True)
                resize_image(outputPathOneImage, outputPathOneImage, size=(512, 512))
                improve_quality(model, outputPathOneImage)
                dataJson['image_iterator'] += 1
        with open(path_to_dataset + "/vars.json", "w") as f:
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

    generate(int(opt.imageCount))
elif int(opt.mode) == 3:
    image_folder = path_to_dataset + "/ResultImages"
    video_name = 'video2.avi'

    images = [img for img in os.listdir(image_folder) if img.startswith("fake")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
