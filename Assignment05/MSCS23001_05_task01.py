# -*- coding: utf-8 -*-
"""HW05Task01_DL_submit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JBpRprxKy2jnkzicOPMkAYWrQ4ORBGFr
"""

"""
###<u> **DEEP LEARNING PROGRAMMING ASSIGNMENT # 5** </u>
* **NAME = HASSAN JAVAID**
* **ROLL NO. = MSCS23001**
* **TASK01 = Training of DCGAN**
* **TASK02 = Implementation of DCGAN as Classifer**
* **ALGORITHM used: Generative Adversarial Networks (GANs)**
"""



# ================================================================
# =========================== Task 01 ============================
# ================================================================

import re
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchsummary import summary
import matplotlib.colors as mcolors
import torchvision.utils as vutils
from tqdm.auto import tqdm

from google.colab import drive
drive.mount('/content/drive')

"""#Function Definitions"""

# Implements plotting of sample images from dataloader
def plotSampleImages(dataloader, suptitle, random_indices, numImages=3):
    dataset = dataloader.dataset
    # random_indices = random.sample(range(len(dataset)), numImages)

    suptitle = f"Sample Images from {suptitle} Dataloader - HCM Malaria Dataset"

    assert numImages%3 == 0     # numImages must be a multiple of 3
    row = numImages // 3
    col = 3
    fig = plt.figure(figsize=(15, row*5))
    # print()
    for i in range(numImages):
        (img, _) = dataset[random_indices[i]]
        img = img.permute(1, 2, 0)
        print(img.shape)
        ax = fig.add_subplot(row, col, i+1)
        # ax.imshow(img)
        img_normalized = img / img.max()
        ax.imshow(img_normalized)
        ax.axis('off')

    fig.suptitle(suptitle, fontsize=16, fontweight='bold', color='blue')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()

def plotGenImages(second_last_real, second_last_gen, img_list, iter, d_loss, g_loss):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot real images
    ax1 = axes[0]
    ax1.axis('off')
    ax1.set_title("Real Images")
    ax1.imshow(np.transpose(vutils.make_grid(second_last_real.to(device)[:BATCH_SIZE],
                                             padding=2, normalize=True).cpu(), (1, 2, 0)))
    ax1.set_aspect('auto')

    # Plot fake images
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title("Fake Images")
    ax2.imshow(np.transpose(img_list[-1].cpu(), (1, 2, 0)))
    ax2.set_aspect('auto')

    fig.suptitle('Iteration: {}, Loss_D: {:.4}, Loss_G: {:.4}'.format(iter+1, d_loss.item(), g_loss.item()))

    plt.tight_layout(pad=1.0, w_pad=1.0, rect=[0, 0, 1, 0.95])

    plt.show()

# Implements saving and loading of the model
def saveModel(model_state_dict, filename):
    # if torch.cuda.is_available():
    #     model.to('cpu')  # Move model to CPU before saving if it's on GPU
    torch.save(model_state_dict, filename)

def loadModel(model, filename):
    if torch.cuda.is_available():
        # Load model on CPU first and then move it to GPU if available
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
        model.to('cuda')
    else:
        model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    return model

"""#Deep Convolutional GAN Model Architecture

###HW5TASK01-32
"""

# Implements the initialization of the weights for DCGAN layers using Xavier method
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)


# Implements discriminator stage downscaling and then BN & LeakyReLU
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=4,
                              stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.bn_check = bn

    def forward(self, x):
        if self.bn_check:
            return self.leakyRelu(self.bn(self.conv(x)))
        else:
            return(self.leakyRelu(self.conv(x)))


# Implements generator stage upscaling and then BN & ReLU
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, firstTranspose=False, tanh=False):
        super().__init__()
        if firstTranspose:
            self.tconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                                     stride=1, padding=0, bias=False)
        else:
            self.tconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                                     stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        if tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

    def forward(self, x1):
        if self.tanh:
            out = self.tanh(self.bn(self.tconv(x1)))
        else:
            out = self.relu(self.bn(self.tconv(x1)))
        return out


# Implements the generator class for our GAN model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_dim = 100
        self.output_dim = 3
        self.initial_dim = 4

        self.upblock1 = UpBlock(self.noise_dim, 128, firstTranspose=True)
        self.upblock2 = UpBlock(128, 64)
        self.upblock3 = UpBlock(64, 32)
        self.upblock4 = UpBlock(32, self.output_dim, tanh=True)


    def forward(self, in_noise):
        x = self.upblock1(in_noise)     # 128x4x4, in_noise=100x1x1
        x = self.upblock2(x)            # 64x8x8
        x = self.upblock3(x)            # 32x16x16
        gen_img = self.upblock4(x)      # 3x32x32 (gen_img)
        return gen_img


# Implements the discriminator class for our GAN model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = 3
        self.output_dim = 1
                                                    # 3x32x32 (input)
        self.downblock1 = DownBlock(3, 32, bn=False)      # 32x16x16
        self.downblock2 = DownBlock(32, 64)     # 64x8x8
        self.downblock3 = DownBlock(64, 128)    # 128x4x4
        # final conv and reshape
        self.conv4 = nn.Conv2d(128, 1, kernel_size=4, stride=1,
                                    padding=0, bias=False) # 1x1x1 (out)
        self.sigmoid = nn.Sigmoid()


    def forward(self, img):
        x1 = self.downblock1(img)       # 32x16x16, img=3x32x32
        x2 = self.downblock2(x1)        # 64x8x8
        x3 = self.downblock3(x2)        # 128x4x4
        x4 = self.conv4(x3)             # 1x1x1 (out)
        check = self.sigmoid(x4)        # 1 (check)
        return check

# # Check device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# # Generate random noise
# # noise = torch.rand(1, 100).to(device)
# noise = torch.randn(100, 1, 1, device = device)

# # GAN goes here
# generator = Generator().to(device)
# generator.apply(weightsInit)
# discriminator = Discriminator().to(device)
# discriminator.apply(weightsInit)

# print(noise.shape)
# # print(generator)
# # print(discriminator)
# summary(generator, noise.shape)
# summary(discriminator, (3, 32, 32))

"""###HW5TASK01-64"""

# Implements the initialization of the weights for DCGAN layers using Xavier method
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)


# Implements discriminator stage downscaling and then BN & LeakyReLU
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_c)
        else:
            self.bn = None

    def forward(self, x):
        if self.bn:
            return self.leakyRelu(self.bn(self.conv(x)))
        else:
            return self.leakyRelu(self.conv(x))


# Implements generator stage upscaling and then BN & ReLU
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, firstTranspose=False, tanh=False):
        super().__init__()
        if firstTranspose:
            # ConvTranspose2d(in_c, out_c, 4, 1, 0)
            self.tconv = nn.ConvTranspose2d(in_c, out_c, 4, 1, 0, bias=False)
        else:
            # else ConvTranspose2d(in_c, out_c, 4, 2, 1)
            self.tconv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        if tanh:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

    def forward(self, x1):
        if self.tanh:
            out = self.tanh(self.tconv(x1))
        else:
            out = self.relu(self.bn(self.tconv(x1)))
        return out


# Implements the generator class for our GAN model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_dim = 100
        self.output_dim = 3
        self.initial_dim = 4

        self.upblock1 = UpBlock(self.noise_dim, 1024, firstTranspose=True)
        self.upblock2 = UpBlock(1024, 512)
        self.upblock3 = UpBlock(512, 256)
        self.upblock4 = UpBlock(256, 128)
        self.upblock5 = UpBlock(128, self.output_dim, tanh=True)

    def forward(self, in_noise):
        x = self.upblock1(in_noise)     # 1024x4x4, in_noise=100x1x1
        x = self.upblock2(x)            # 512x8x8
        x = self.upblock3(x)            # 256x16x16
        x = self.upblock4(x)            # 128x32x32
        gen_img = self.upblock5(x)       # 3x64x64 (gen_img)
        return gen_img


# Implements the discriminator class for our GAN model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = 3
        self.downblock1 = DownBlock(self.input_dim, 64, bn=False) # 64x32x32, 3x64x64 (input)
        self.downblock2 = DownBlock(64, 128)     # 128x16x16
        self.downblock3 = DownBlock(128, 256)     # 256x8x8
        self.downblock4 = DownBlock(256, 512)    # 512x4x4
        # final conv and reshape
        self.conv4 = nn.Conv2d(512, 1, 4, 1, 0, bias=False) # 1x1x1 (out)
        self.sigmoid = nn.Sigmoid()


    def get_layer(self, layer_idx):
        if layer_idx == 0:
            return self.downblock1
        elif layer_idx == 1:
            return self.downblock2
        elif layer_idx == 2:
            return self.downblock3
        elif layer_idx == 3:
            return self.downblock4
        elif layer_idx == 4:
            return self.conv4
        else:
            raise IndexError("Layer index out of range")

    def forward(self, img):
        x1 = self.downblock1(img)       # 128x32x32, img=3x64x64
        x2 = self.downblock2(x1)        # 256x16x16
        x3 = self.downblock3(x2)        # 512x8x8
        x4 = self.downblock4(x3)        # 1024x4x4
        x5 = self.conv4(x4)             # 1x1x1 (out)
        check = self.sigmoid(x5)        # 1 (check)

        return check

"""#DataLoader setup"""

# Define paths
ROOT_DIR = '/content/drive/MyDrive/400/HCM'
train_folder = ROOT_DIR + '/' + 'train'
val_folder = ROOT_DIR + '/' + 'val'
test_folder = ROOT_DIR + '/' + 'test'

# Custom dataset class for HCM Datasets
class HCMDataset(Dataset):
    def __init__(self, data, folder_path, resize, label):
        self.data = data
        self.folder_path = folder_path
        if resize:
            self.transform = transforms.Compose([
                            transforms.Resize(IMAGE_SIZE),
                            transforms.CenterCrop(IMAGE_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])
        else:
            self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        image = Image.open(os.path.join(self.folder_path, img_name))
        if self.transform:
            image = self.transform(image)
        return (image, img_name)


# Implements creation of Task-01 datasets from their respective folders
def createDataset(folder_path, resize=True, label=None):
    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    return HCMDataset(image_files, folder_path, resize, label)

# Define hyper parameters
BATCH_SIZE = 32
IMAGE_SIZE = 64
LEARNING_RATE = 0.0002
RANDOM_SEED = 42

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Prepare datasets for train, valid and test datasets
train_dataset = createDataset(train_folder, resize=True)
val_dataset = createDataset(val_folder, resize=True)
test_dataset = createDataset(test_folder, resize=True)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Valid dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")
print()

# Prepare Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Without usng transform
t_dataset = createDataset(train_folder, resize=False)
t_loader = DataLoader(t_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
numImages = 3
random_indices = random.sample(range(len(t_dataset)), numImages)

# Plot sample images from dataloaders
plotSampleImages(train_loader, "Train", random_indices, numImages)
plotSampleImages(t_loader, "Train without resize", random_indices, numImages)

"""#GAN initialization and Model setup"""

# # Create the discriminator
discriminator = Discriminator().to(device)
discriminator.apply(initialize_weights)


# # Create the generator
generator = Generator().to(device)
generator.apply(initialize_weights)

# Generate random noise
noise = torch.randn(100, 1, 1, device = device)

summary(generator, noise.shape)
summary(discriminator, (3, 64, 64))


beta = 0.5

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE,
                                 betas=(beta, 0.999))
optimizer_Disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE,
                                  betas=(beta, 0.999))

# Check Generator output
noise_input = torch.randn(BATCH_SIZE, 100, 1, 1, device = device)
generator.eval()
generated_image = generator(noise_input)
print(generated_image.shape)
plt.imshow(generated_image[0][0].cpu().detach().numpy())
plt.show()

# # Load the model again to continue training
# DISC_LOAD_PATH = ROOT_DIR + '/' + "Task01/progress_discriminator_20.pth"
# GEN_LOAD_PATH = ROOT_DIR + '/' + "Task01/progress_generator_20.pth"
# # Load pre-trained discriminator and generator
# loaded_discriminator = loadModel(discriminator, DISC_LOAD_PATH)
# loaded_generator = loadModel(generator, GEN_LOAD_PATH)

"""#Model Training"""

# Training loop
epochs = 30
LATENT_DIM = 100
gen_loss_history = []
disc_loss_history = []
img_list = []
count = 0
iter = 0
batch = 0
plot_iter = 250

for epoch in tqdm(range(epochs)):
    for i, (imgs, _) in tqdm(enumerate(train_loader, 0)):
        count +=imgs.shape[0]
        valid = torch.ones(imgs.shape[0], device=device).requires_grad_(False)
        fake = torch.zeros(imgs.shape[0], device=device).requires_grad_(False)

        real_imgs = imgs.to(device)

        # ======== Discriminator training ============
        optimizer_Disc.zero_grad()

        # Training with real batch
        disc_r = discriminator(real_imgs).view(-1)                 # D(x)
        real_loss = criterion(disc_r, valid)                       # log(D(x))
        real_loss.backward()

        # Training with fake batch
        noise_input = torch.randn(imgs.shape[0], LATENT_DIM, 1, 1, device = device)
        generator_imgs = generator(noise_input)
        disc_f_fake = discriminator(generator_imgs.detach()).view(-1)            # 1-D(G(z))

        fake_loss = criterion(disc_f_fake, fake)                        # log(1-D(G(z)))
        fake_loss.backward()

        # Maximize log(D(x)) + log(1-D(G(z)))
        d_loss = real_loss + fake_loss
        optimizer_Disc.step()

        # ======== Generator training ================
        optimizer_Gen.zero_grad()
        disc_f = discriminator(generator_imgs).view(-1)
        g_loss = criterion(disc_f, valid)

        g_loss.backward()
        optimizer_Gen.step()


        # Print training status
        if iter % 100 == 0:
          print(f"[{epoch+1}/{epochs}] [{i+1}/{len(train_loader)}] [Iter: {iter+1}]\tLoss_D: {d_loss.item():.4f}\tLoss_G: {g_loss.item():.4f}")

        # Store generator and discriminator losses
        gen_loss_history.append(g_loss.item())
        disc_loss_history.append(d_loss.item())

        # Display generator progress
        if iter % plot_iter == 0:
            print('Iter: {}, Loss_D: {:.4}, Loss_G:{:.4}'.format(iter,d_loss.item(), g_loss.item()))
            print()
            img_list.append(vutils.make_grid(generator_imgs, padding=2, normalize=True))
            plotGenImages(real_imgs, generator_imgs, img_list, iter, d_loss, g_loss)

        # Increment iteration count
        iter += 1

"""#Save the models"""

# save the model for future use
DISC_SAVE_PATH = ROOT_DIR + '/' + "Task01/actual_1500_progress_discriminator_60.pth"
GEN_SAVE_PATH = ROOT_DIR + '/' + "Task01/actual_1500_progress_generator_60.pth"
torch.save(discriminator.state_dict(), DISC_SAVE_PATH)
torch.save(generator.state_dict(), GEN_SAVE_PATH)

print(len(img_list))
print(img_list[0].shape)

print(generator_imgs.shape)

"""#Plot Loss Curve"""

plt.figure(figsize=(10, 10))
plt.title("Loss Graph")
plt.plot(gen_loss_history, label="Generator Loss")
plt.plot(disc_loss_history, label="Discriminator")
plt.ylabel("Loss")
plt.xlabel("Iterators")
plt.legend()
plt.show()

# End of code