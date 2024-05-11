import os

import torch
import torch.nn as nn
from torchvision.utils import save_image

from seed_walker_psp import SeedWalkerPsp

import argparse
parser = argparse.ArgumentParser(description="midiGANn")
parser.add_argument("--dataset_path", type=str, default='datasets/twimg_64x64/', help="path of the target")
parser.add_argument("--test", dest="test", action="store_true", help="if true, only use test")
parser.add_argument("--train", dest="train", action="store_true", help="if true, only use cpu")
parser.add_argument("--play", dest="play", action="store_true", help="if true, input image needs no pre-procssing")
parser.add_argument("--number", type=int, default=4, help="output number of multi-modal translation")
args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Hyperparameter settings
"""
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 64
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
# Number of training epochs
num_epochs = 100
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

loss = nn.BCELoss()



import torch
import torch.nn as nn
"""
Network Architectures
The following are the discriminator and generator architectures
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)






import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
# from model import discriminator, generator
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import glob
import cv2
import torchvision
avgpool = nn.AdaptiveAvgPool2d((128, 128))
from PIL import Image


"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
# G = generator().to(device)
# D = discriminator().to(device)
#
# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

"""
Image transformation and dataloader creation
Note that we are training generation and not classification, and hence
only the train_loader is loaded
"""
# Transform
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

avgpool = nn.AdaptiveAvgPool2d((64, 64))

# Root directory for dataset
dataroot = "data/celeba"
dataset = 'datasets/twimg_64/'

class afhqDataset(Dataset):
    def __init__(self, root_dir=dataset):
        self.root_dir = root_dir
        self.img_dir = root_dir + '*.png'

    def __len__(self):
        return len(glob.glob(self.img_dir))

    def __getitem__(self, idx):
        file_name = glob.glob(self.img_dir)[idx]
        # print(file_name)
        image = cv2.imread(file_name)/255.
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)
        image = avgpool(image)
        return image

train_set = afhqDataset()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

image = cv2.imread(glob.glob(dataset + '/*')[1])/255.
image = torch.Tensor(image)
image = image.permute(2, 0, 1)
image = avgpool(image)
image = image.permute(1, 2, 0)
plt.imshow(image)

# Create the generator
netG = Generator(ngpu).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
#print(netG)

netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
# print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)



"""
Train

"""
if args.train:

    for epoch in range(num_epochs):
        for idx, (imgs) in enumerate(train_loader):
            idx += 1

            # Training the discriminator
            # Real inputs are actual images of the MNIST dataset
            # Fake inputs are from the generator
            # Real inputs should be classified as 1 and fake as 0
            real_inputs = imgs.to(device)
            real_outputs = netD(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)

            noise = (torch.rand(real_inputs.shape[0], nz, 1, 1) - 0.5) / 0.5
            noise = noise.to(device)
            # print(noise.shape)
            fake_inputs = netG(noise)
            fake_outputs = netD(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

            outputs = torch.cat((real_outputs.view(-1).unsqueeze(1), fake_outputs.view(-1).unsqueeze(1)), 0)
            targets = torch.cat((real_label, fake_label), 0)

            D_loss = loss(outputs, targets)
            optimizerD.zero_grad()
            D_loss.backward()
            optimizerD.step()

            # Training the generator
            # For generator, goal is to make the discriminator believe everything is 1
            noise = (torch.rand(real_inputs.shape[0], nz, 1, 1)-0.5)/0.5
            noise = noise.to(device)
            #print(noise.shape)
            fake_inputs = netG(noise)
            fake_outputs = netD(fake_inputs)
            fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
            G_loss = loss(fake_outputs.view(-1).unsqueeze(1), fake_targets)
            optimizerG.zero_grad()
            G_loss.backward()
            optimizerG.step()

            if idx % 10 == 0 or idx == len(train_loader):
                print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))

        if (epoch+1) % 20 == 0:
            print("save models")
            # Save the model checkpoints
            torch.save(netG.state_dict(), 'train/color_gan_G.ckpt')
            torch.save(netD.state_dict(), 'train/color_gan_D.ckpt')
        sample_dir = "train/"
        # Save real images
        if (epoch+1) == 1:
            images = imgs[0]#.reshape(imgs, 1, image_size, image_size)
            save_image(images, os.path.join(sample_dir, 'real_images.png'))
        #    print("save", os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake_inputs#.reshape(fake_inputs.size(0), 1, image_size, image_size)
    save_image(denorm(fake_images[0]), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
    # print("save",  os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))



"""
NOTE!: Play

"""
if args.play:
    import time
    import torchvision.transforms as T
    from PIL import ImageFilter
    transformim = T.ToPILImage()
    netG.load_state_dict(torch.load('train/color_gan_G.ckpt'))
    netG.eval()

    seed_walker = SeedWalkerPsp(100)

    seed = 0
    frame = 0
    pTime = 0
    #torch.Size([64, 100, 1, 1])
    while True:

        seed_walker.update(frame)

        z = (torch.rand(1, nz, 1, 1) - 0.5) / 0.5
        z = seed_walker.calc_tensor(z)
        z = z.to(device)

        generated = netG(z)
        generated = generated.cpu().detach()
        img = transformim(generated[0])

        frame += 1
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        size = 512

        img = img.filter(ImageFilter.BLUR)

        img = np.array(img.resize((size, size), Image.LANCZOS))
        cv2.putText(img, f'FPS: {int(fps)} {int(frame)} seed: {seed_walker.get_seed()}', (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:
            break


"""
NOTE!: Test

"""
if args.test:
    print("test")
