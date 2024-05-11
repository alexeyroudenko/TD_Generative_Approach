
print("create ext")
import os 
import sys
sys.path.append(".")
sys.path.append("Z:\\Enviroments\\ml\\Lib\\site-packages")
os.environ["PATH"] += os.pathsep + f'Z:\\Enviroments\\ml\\DLLs'
os.environ["PATH"] += os.pathsep + f'Z:\\Enviroments\\ml\\\\Library\\bin'

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# from TDStoreTools import StorageManager
from seed_walker_psp import SeedWalkerPsp
 
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import transforms
# import numpy as np
# import cv2
# avgpool = nn.AdaptiveAvgPool2d((64, 64))
# from PIL import Image
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
Hyperparameter settings
"""
dataroot = "data/celeba"
workers = 2
batch_size = 64
image_size = 64
nc = 3  # Number of channels in the training images. For color images this is 3
nz = 100    # Size of z latent vector (i.e. size of generator input)
ngf = 64    # Size of feature maps in generator
ndf = 64    # Size of feature maps in discriminator
num_epochs = 1000 # Number of training epochs
lr = 0.0002 # Learning rate for optimizers
beta1 = 0.5 # Beta1 hyperparam for Adam optimizers
ngpu = 1    # Number of GPUs available. Use 0 for CPU mode.
loss = nn.BCELoss()
latent_size = 100


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

class ColorGANExt:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        storedItems = [
            {'name': 'State', 'default': 0, 'readOnly': False,'property': True, 'dependable': True},
            {'name': 'StoredProperty', 'default': 0, 'readOnly': False,'property': True, 'dependable': True},
            {'name': 'CurrentFrame','default': 0, 'readOnly': False,'property': True, 'dependable': True},
            {'name': 'CountFrames','default': 0, 'readOnly': False,'property': True, 'dependable': True},
            {'name': 'CountErrors', 'default': 0, 'readOnly': False,'property': True, 'dependable': True},
            {'name': 'CurrentSession','default': "20230315-", 'readOnly': False,'property': True, 'dependable': True},
            {'name': 'FacedFrame','default': 0,'readOnly': False,'property': True, 'dependable': True},
            {'name': 'CurrentSession','default': "20200220",'readOnly': False,'property': True, 'dependable': True},
            {'name': 'OutFilename','default': "20200220.mp4",'readOnly': False,'property': True, 'dependable': True},
        ]

        """
        Image transformation and dataloader creation
        Note that we are training generation and not classification, and hence
        only the train_loader is loaded
        """
        # Transform
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        avgpool = nn.AdaptiveAvgPool2d((64, 64))
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
        #print(netD)
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

        # """
        # NOTE!: Play
        # """
        import time
        import torchvision.transforms as T
        from PIL import ImageFilter
        transformim = T.ToPILImage()
        
        model_path = op('model').text #'models/gan_color_G.ckpt'
        netG.load_state_dict(torch.load(model_path))
        netG.eval()
        self.netG = netG
        self.Image = None
        self.one = 1
        self.img = None
        print("ColorGAN init done")
        self.seed_walker = SeedWalkerPsp(100)

    def Generate(self):
        z_array = op('latent_vector').numpyArray()
        z = torch.tensor(z_array).reshape(1, 100, 1,  1)
        z = z.to(device)
        
        import torchvision.transforms as T
        transformim = T.ToPILImage()
        generated = self.netG(z)
        generated = generated.cpu().detach()
        img = transformim(generated[0])
        img.save("tmp.jpg")        
        # torch.permute(img, (2, 0, 1))
        # self.img = np.array(generated)
        # return self.img