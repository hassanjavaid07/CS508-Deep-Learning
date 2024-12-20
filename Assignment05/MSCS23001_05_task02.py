# -*- coding: utf-8 -*-
"""HW05Task02_DL_submit.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cvFFrvTvSAZdK8p7CZl-ZMlA5CNRy2Vc
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
# =========================== Task 02 ============================
# ================================================================

import re
import os
import sys
import cv2
import csv
import codecs
import random
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from os import listdir
from os.path import isfile, join
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
from sklearn.metrics import accuracy_score
from torch.autograd import Function
import torchvision.utils as vutils

from google.colab import drive
drive.mount('/content/drive')

"""#Function Definitions"""

# Implements plotting of sample images from dataloader
def plotSampleImages(dataloader, suptitle, random_indices, classes, numImages=3):
    dataset = dataloader.dataset
    # random_indices = random.sample(range(len(dataset)), numImages)

    suptitle = f"Sample Images from {suptitle} Dataloader - LCM Malaria Dataset"

    assert numImages%3 == 0     # numImages must be a multiple of 3
    row = numImages // 3
    col = 3
    fig = plt.figure(figsize=(15, row*5))
    for i in range(numImages):
        (img, label, _) = dataset[random_indices[i]]
        img = img.permute(1, 2, 0)
        print(img.shape)
        ax = fig.add_subplot(row, col, i+1)
        ax.set_title(f"Class Name: {classes[label]}, Label: {label}", fontsize=14)
        ax.imshow(img)
        ax.axis('off')

    fig.suptitle(suptitle, fontsize=16, fontweight='bold', color='blue')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()

# Visualize correct and wrong predictions
# Implements visualization of test predictions
def visualizePredictions(test_loader, all_preds, all_labels):
    correct_indices = np.where(np.array(all_preds) == np.array(all_labels))[0]
    wrong_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
    correct_samples = random.sample(list(correct_indices), min(5, len(correct_indices)))
    wrong_samples = random.sample(list(wrong_indices), min(5, len(wrong_indices)))
    title = "Correct Predictions"
    print("Correct Predictions:")
    plotSamples(test_loader, correct_samples, all_preds, title)
    title = "Wrong Predictions"
    print("Wrong Predictions:")
    plotSamples(test_loader, wrong_samples, all_preds, title)



def plotSamples(test_loader, indices, all_preds, title):
    images = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for i in indices:
        _, _, image_name = test_loader.dataset[i]
        image = Image.open(os.path.join(test_folder, image_name))
        img_processed = transform(image).unsqueeze(0) # Add batch dimension
        images.append(img_processed)
    preds = [all_preds[i] for i in indices]
    images = torch.stack(images)
    plt.figure(figsize=(10, 5))
    plt.suptitle(title)
    for i in range(len(indices)):
        plt.subplot(2, 5, i + 1)
        input_np = images[i][0].squeeze().detach().cpu().numpy()
        plt.imshow(input_np.transpose(1, 2, 0))
        plt.title(f'Predicted Label: {preds[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.draw()


# Test Function to evaluate
# Implements the test function to evaluate our trained model
def evaluateModel(trained_G, trained_D, test_loader):
    trained_G.eval()
    trained_D.eval()
    # classes = list(range(10))
    classes_r = {1:"gametocyte", 2:'schizont', 3:'trophozoite', 4:'ring'}
    classes = list(classes_r.keys())
    NUM_CLASSES = len(classes_r.keys())
    # test_history = {'accuracy': [], 'loss': []}
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for images, labels, _ in test_loader:

            real_imgs, labels = images.to(device), labels.to(device).float()

            valid = torch.ones(images.shape[0], device=device).requires_grad_(False)
            fake = torch.zeros(images.shape[0], device=device).requires_grad_(False)

            noise_input = torch.randn(images.shape[0], LATENT_DIM, 1, 1, device=device)
            generator_imgs = trained_G(noise_input)
            disc_r, _ = trained_D(real_imgs)
            real_loss = adv_loss(disc_r.view(-1), valid)

            disc_f, preds = trained_D(generator_imgs.detach())
            preds = torch.argmax(preds, dim=1).float()
            fake_loss = adv_loss(disc_f.view(-1), fake)

            d_loss = real_loss + fake_loss
            mc_loss = clf_loss(preds, labels)
            total_loss = d_loss + mc_loss

            # _, preds = model(images)
            # _, preds = torch.max(preds, 1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f} - Total Loss: {total_loss:.4f}")

    return accuracy, total_loss, all_preds, all_labels

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

# Load ground truth labels from xml files in annotation_folder
XML_EXT = '.xml'
ENCODE_METHOD = "UTF8"


class PascalVocReader:
    def __init__(self, file_path):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.file_path = file_path
        self.verified = False
        try:
            self.parse_xml()
        except:
            pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, bnd_box, difficult):
        x_min = int(float(bnd_box.find('xmin').text))
        y_min = int(float(bnd_box.find('ymin').text))
        x_max = int(float(bnd_box.find('xmax').text))
        y_max = int(float(bnd_box.find('ymax').text))
        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def parse_xml(self):
        assert self.file_path.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xml_tree = ElementTree.parse(self.file_path, parser=parser).getroot()
        filename = xml_tree.find('filename').text
        try:
            verified = xml_tree.attrib['verified']
            if verified == 'yes':
                self.verified = True
        except KeyError:
            self.verified = False

        for object_iter in xml_tree.findall('object'):
            bnd_box = object_iter.find("bndbox")
            label = object_iter.find('name').text
            difficult = False
            if object_iter.find('difficult') is not None:
                difficult = bool(int(object_iter.find('difficult').text))
            self.add_shape(label, bnd_box, difficult)
        return True

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

# # ==============================================================================
# # ========================== DO NOT DELETE THIS ================================
# # ==============================================================================


# def createLabelCSV(annotation_folder, csv_file_path):
#     print(f"Creating Label CSV file at: {csv_file_path}")
#     files=[f for f in listdir(annotation_folder)if isfile(join(annotation_folder,f))]
#     classes = {"gametocyte": 1, 'schizont': 2,'trophozoite': 3,'ring': 4}
#     class_names = []
#     class_labels = []
#     img_names = []
#     for f in files:
#         xml_path=join(annotation_folder,f)
#         img_name=f.split('x')[0]+'x.png'
#         # img_path=join(images,img_name)
#         img_names.append(img_name)
#         t_voc_parse_reader = PascalVocReader(xml_path)
#         shapes = t_voc_parse_reader.get_shapes()
#         for s in shapes:
#             name = s[0]
#             class_names.append(name)
#             class_labels.append(classes[name])


#     # Create a new CSV file and write the data
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Filename', 'Class Label', 'Class Name'])
#         for img_name, label, name in zip(img_names, class_labels, class_names):
#             writer.writerow([img_name, label, name])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None

    def forward(self, x):
        self.feature_maps = []
        self.gradient = []

        # Define hooks to store the feature maps and gradients
        def save_feature_maps(module, input, output):
            self.feature_maps.append(output)

        def save_gradient(module, grad_input, grad_output):
            self.gradient.append(grad_output[0])

        hooks = []
        for layer_idx in range(self.target_layer + 1):
            module = self.model.get_layer(layer_idx)
            hooks.append(module.register_forward_hook(save_feature_maps))
        hooks.append(self.model.get_layer(self.target_layer).register_backward_hook(save_gradient))

        # Forward pass
        self.model(x)
        output = self.model(x)

        # Backward pass
        self.model.zero_grad()
        output[0].backward()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return output, self.feature_maps, self.gradient

    def __call__(self, x):
        return self.forward(x)

def getGradCAMObj(model, target_layer):
    return GradCAM(model, target_layer)


# Implements overlay CAM on the input image
def getCAMOverlay(input_image, cam):
    # Normalize CAM
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (input_image.shape[1], input_image.shape[0]))

    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay heatmap on the input image
    overlaid_img = np.float32(heatmap) / 255 + np.float32(input_image)
    overlaid_img = overlaid_img / np.max(overlaid_img)

    return overlaid_img

# Implements functionlity for real-time image preprocessing and class prediction
def preprocessImage(test_image_path):
    img = Image.open(test_image_path)

    # Define transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),                  # Resize
        transforms.ToTensor(),                        # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize the pixel values
    ])

    # Apply transformations to the image
    img_t = transform(img).unsqueeze(0)

    return img_t

def predictInputImage(noise_input, trained_G, trained_D):
    trained_G.eval()
    trained_D.eval()

    # Perform prediction
    with torch.no_grad():
        generator_imgs = trained_G(noise_input)
        disc_f, preds = trained_D(generator_imgs.detach())
        predicted = torch.argmax(preds, dim=1).float()

    return predicted

"""#Paths and DataLoader Setup"""

# Define paths
ROOT_DIR = '/content/drive/MyDrive/Assignment05/400/HCM'
train_folder = ROOT_DIR + '/' + 'train'
val_folder = ROOT_DIR + '/' + 'val'
test_folder = ROOT_DIR + '/' + 'test'

# For label reading - HCM IMAGES
LABEL_FOLDER = '/content/drive/MyDrive/Assignment05/400/Label_Info'
ANNOTATION_FOLDER = '/content/drive/MyDrive/Assignment05/400/Annotation'

TRAIN_LABEL_FILENAME = 'trainHCM_labelinfo.csv'
train_annotation_folder = ANNOTATION_FOLDER + '/' + 'HCM/train'
train_label_folder = LABEL_FOLDER + '/' + 'HCM/train'
train_csv_file_path = train_label_folder + '/' + TRAIN_LABEL_FILENAME

VAL_LABEL_FILENAME = 'valHCM_labelinfo.csv'
val_annotation_folder = ANNOTATION_FOLDER + '/' + 'HCM/val'
val_label_folder = LABEL_FOLDER + '/' + 'HCM/val'
val_csv_file_path = val_label_folder + '/' + VAL_LABEL_FILENAME

TEST_LABEL_FILENAME = 'testHCM_labelinfo.csv'
test_annotation_folder = ANNOTATION_FOLDER + '/' + 'HCM/test'
test_label_folder = LABEL_FOLDER + '/' + 'HCM/test'
test_csv_file_path = test_label_folder + '/' + TEST_LABEL_FILENAME

# Create Dataset Class
# Custom dataset class for HCM Datasets
class HCMDataset(Dataset):
    def __init__(self, data, labels, folder_path, resize):
        self.data = data
        self.labels = labels
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
        label = self.labels[idx]
        image = Image.open(os.path.join(self.folder_path, img_name))
        if self.transform:
            image = self.transform(image)
        return (image, label, img_name)


# Implements creation of Task-01 datasets from their respective folders
def createDataset(folderPath, labelFilePath, resize=True):
    labels_df = pd.read_csv(labelFilePath)
    labels = []
    image_files = []
    for idx, row in labels_df.iterrows():
        filename = row['Filename']
        label = int(row['Class Label'])
        image_files.append(filename)
        labels.append(label)
    return HCMDataset(image_files, labels, folderPath, resize)

# Define hyper parameters
BATCH_SIZE = 32
IMAGE_SIZE = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.0002
RANDOM_SEED = 42
LATENT_DIM = 100

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


classes_r = {1:"gametocyte", 2:'schizont', 3:'trophozoite', 4:'ring'}
NUM_CLASSES = len(classes_r.keys())
print(NUM_CLASSES)

# Create Label csv files for our GAN images (only required to run one time)
# createLabelCSV(train_annotation_folder, train_csv_file_path)
# createLabelCSV(val_annotation_folder, val_csv_file_path)
# createLabelCSV(test_annotation_folder, test_csv_file_path)

# Prepare datasets for train, valid and test datasets
train_dataset = createDataset(train_folder, train_csv_file_path, resize=True)
val_dataset = createDataset(val_folder, val_csv_file_path, resize=True)
test_dataset = createDataset(test_folder, test_csv_file_path, resize=True)

print(f"Train dataset length: {len(train_dataset)}")
print(f"Valid dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")
print()

# Prepare Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

sample = train_dataset[0]
sample[1]

# Without using transform
t_dataset = createDataset(train_folder, train_csv_file_path, resize=False)
t_loader = DataLoader(t_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
numImages = 3
random_indices = random.sample(range(len(t_dataset)), numImages)

# Plot sample images from dataloaders
plotSampleImages(train_loader, "Train", random_indices, classes_r, numImages)
plotSampleImages(t_loader, "Train without resize", random_indices, classes_r, numImages)

"""#Deep Convolutional GAN Model Architecture with Classifier"""

# Implements the initialization of the weights for DCGAN layers with xavier method
def xavierInitialization(m):
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
            self.tconv = nn.ConvTranspose2d(in_c, out_c, 4, 1, 0, bias=False)
        else:
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
    def __init__(self, num_classes, dropout_rate):
        super().__init__()

        self.input_dim = 3
        self.output_dim = 1
                                                  # 3x64x64 (input)
        self.downblock1 = DownBlock(3, 64, bn=False)       # 64x32x32
        self.downblock2 = DownBlock(64, 128)     # 128x16x16
        self.downblock3 = DownBlock(128, 256)     # 256x8x8
        self.downblock4 = DownBlock(256, 512)    # 512x4x4
        # final conv and reshape
        self.conv4 = nn.Conv2d(512, 1, 4, 1, 0, bias=False) # 1x1x1 (out)
        self.sigmoid = nn.Sigmoid()

        # Multi-label classifier layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 4 * 4, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        # self.softmax = F.softmax()

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

        # Multilabel classifier stage
        x = self.flatten(x4)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return (check, x)

"""#GAN initialization and Model setup"""

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Generate random noise
noise = torch.randn(100, 1, 1, device = device)

# GAN goes here
generator = Generator().to(device)
generator.apply(xavierInitialization)
discriminator = Discriminator(NUM_CLASSES, DROPOUT_RATE).to(device)
discriminator.apply(xavierInitialization)

print(noise.shape)
summary(generator, noise.shape)
summary(discriminator, (3, 64, 64))

beta = 0.5

# Define loss functions and optimizers
adv_loss = nn.BCELoss()
clf_loss = nn.CrossEntropyLoss()
optimizer_Gen = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE,
                                 betas=(beta, 0.999))
optimizer_Disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE,
                                  betas=(beta, 0.999))

noise_input = torch.randn(BATCH_SIZE, 100, 1, 1, device = device)
generator.eval()
generated_image = generator(noise_input)
print(generated_image.shape)
plt.imshow(generated_image[0][0].cpu().detach().numpy(), )
plt.show()

"""#Model Training"""

# Training GAN
epochs = 30
LATENT_DIM = 100
gen_loss_history = []
disc_loss_history = []
train_acc_history = []
img_list = []

val_gen_loss_history = []
val_disc_loss_history = []
val_acc_history = []

count = 0
iter = 0
plot_iter = 250

for epoch in tqdm(range(epochs)):
    total_samples = 0
    correct_predictions = 0
    total_val_samples = 0
    correct_val_predictions = 0

    # Training Loop
    generator.train()
    discriminator.train()
    for i, (imgs, labels, _) in tqdm(enumerate(train_loader, 0)):
        count +=imgs.shape[0]
        valid = torch.ones(imgs.shape[0], device=device).requires_grad_(False)
        fake = torch.zeros(imgs.shape[0], device=device).requires_grad_(False)

        real_imgs = imgs.to(device)
        labels = labels.to(device).float()

        # ======== Discriminator training ============
        optimizer_Disc.zero_grad()

        # Training with real batch
        disc_r, _ = discriminator(real_imgs)                 # D(x)
        real_loss = adv_loss(disc_r.view(-1), valid)         # log(D(x))
        real_loss.backward()

        # Training with fake batch
        noise_input = torch.randn(imgs.shape[0], LATENT_DIM, 1, 1, device = device)
        generator_imgs = generator(noise_input)
        disc_f_fake, preds = discriminator(generator_imgs.detach())   # 1-D(G(z))
        fake_loss = adv_loss(disc_f_fake.view(-1), fake)                        # log(1-D(G(z)))
        fake_loss.backward()

        preds = torch.argmax(preds, dim=1).float()

        # Maximize log(D(x)) + log(1-D(G(z)))
        d_loss = real_loss + fake_loss
        mc_loss = clf_loss(preds, labels)
        total_d_loss = d_loss + mc_loss

        optimizer_Disc.step()

        # ======== Generator training ================
        optimizer_Gen.zero_grad()

        disc_f, _ = discriminator(generator_imgs)
        g_loss = adv_loss(disc_f.view(-1), valid)
        g_loss.backward()

        optimizer_Gen.step()

        total_samples += labels.size(0)
        correct_predictions += (preds == labels).sum().item()

        # Print status
        if i % 100 == 0:
            print("Training Loop")
            print(f"[Epoch: {epoch+1}/{epochs}] [{i+1}/{len(train_loader)}] [Iter: {iter+1}]\tTotal Loss_D: {total_d_loss.item():.4f}\tClf_loss: {mc_loss.item():.4f}\tLoss_D: {d_loss.item():.4f}\tLoss_G: {g_loss.item():.4f}")

        # Store generator and discriminator losses
        gen_loss_history.append(g_loss.item())
        disc_loss_history.append(d_loss.item())

        # Display generator progress
        if iter % plot_iter == 0:
            print('Iter: {}, Loss_D: {:.4}, Loss_G:{:.4}'.format(iter,d_loss.item(), g_loss.item()))
            print()
            img_list.append(vutils.make_grid(generator_imgs, padding=2, normalize=True))
            plotGenImages(real_imgs, generator_imgs, img_list, iter, d_loss, g_loss)

        iter +=1

    print(f"Total Train Samples = {total_samples}")
    train_accuracy = correct_predictions / total_samples
    train_acc_history.append(train_accuracy)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print()

    # Validation loop
    generator.eval()
    discriminator.eval()
    with torch.no_grad():


        for i, (imgs, labels, _) in enumerate(val_loader, 0):
            real_imgs = imgs.to(device)
            labels = labels.to(device).float()

            valid = torch.ones(imgs.shape[0], device=device).requires_grad_(False)
            fake = torch.zeros(imgs.shape[0], device=device).requires_grad_(False)

            # ======== Discriminator validation ============
            noise_input = torch.randn(imgs.shape[0], LATENT_DIM, 1, 1, device=device)
            generator_imgs = generator(noise_input)
            disc_r, _ = discriminator(real_imgs)
            real_loss = adv_loss(disc_r.view(-1), valid)

            disc_f, val_preds = discriminator(generator_imgs.detach())
            val_preds = torch.argmax(val_preds, dim=1).float()
            fake_loss = adv_loss(disc_f.view(-1), fake)

            # Maximize log(D(x)) + log(1-D(G(z))) for discriminator validation loss
            d_loss = real_loss + fake_loss
            mc_loss = clf_loss(val_preds, labels)
            total_d_loss = d_loss + mc_loss

            # ======== Generator validation ================
            disc_f, _ = discriminator(generator_imgs)
            g_loss = adv_loss(disc_f.view(-1), valid)

            # Store generator & discriminator validation loss
            val_gen_loss_history.append(g_loss.item())
            val_disc_loss_history.append(total_d_loss.item())

            # Calculate validation accuracy
            total_val_samples += labels.size(0)
            correct_val_predictions += (val_preds == labels).sum().item()
            # Print status
            print("Validation loop")
            print(f"[Epoch: {epoch+1}/{epochs}] [{i+1}/{len(val_loader)}]\tTotal_Loss_D: {total_d_loss.item():.4f}\tClf_loss: {mc_loss.item():.4f}\tLoss_D: {d_loss.item():.4f}\tLoss_G: {g_loss.item():.4f}")


    print(f"Total Validation Samples = {total_val_samples}")
    val_accuracy = correct_val_predictions / total_val_samples
    val_acc_history.append(val_accuracy)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print()

"""#Save the Model"""

# save the model for future use
DISC_SAVE_PATH = ROOT_DIR + '/' + "Task02/actual_800_new_discriminator_30.pth"
GEN_SAVE_PATH = ROOT_DIR + '/' + "Task02/actual_800_new_generator_30.pth"
saveModel(discriminator.state_dict(), DISC_SAVE_PATH)
saveModel(generator.state_dict(), GEN_SAVE_PATH)

"""#GradCAM Visualization"""

DISC_LOAD_PATH = ROOT_DIR + '/' + "Task02/new_discriminator_60.pth"

# Load your pre-trained model state dict
discriminator = Discriminator(num_classes=4, dropout_rate=0.2)
loaded_discriminator = loadModel(discriminator, DISC_LOAD_PATH)

# Initialize GradCAM - Target layer is the 2nd last convolutional layer
gradcam = getGradCAMObj(loaded_discriminator, target_layer=2)
loaded_discriminator.eval()


# Pass input image through the GradCAM model
input_image, _, img_name = train_dataset[91]
image = Image.open(os.path.join(train_folder, img_name))
transform = transforms.ToTensor()
image = transform(image)
input_image = input_image.unsqueeze(dim=0)
output, feature_maps, gradients = gradcam(input_image)


# Compute importance weights, weighted sum, apply final ReLU
weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
cam = torch.sum(weights * feature_maps[-1], dim=1, keepdim=True)
cam = torch.relu(cam)


# Overlay CAM on input image
input_image_np = image.squeeze().detach().cpu().numpy()
cam_np = cam.squeeze().detach().cpu().numpy()
print(input_image_np.shape)
overlaid_image = getCAMOverlay(input_image_np.transpose(1, 2, 0), cam_np)

# Visualize results
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(input_image_np.transpose(1, 2, 0))
plt.title('Original Image')
plt.axis('off')

# CAM Heatmap
plt.subplot(1, 3, 2)
plt.imshow(cam_np, cmap='jet')
plt.title('CAM Heatmap')
plt.axis('off')

# CAM Overlay
plt.subplot(1, 3, 3)
plt.imshow(overlaid_image)
plt.title('CAM Overlay')
plt.axis('off')

plt.show()

print(train_acc_history)
print(len(img_list))
print(img_list[0].shape)

print(generator_imgs.shape)

"""#Plot Losses"""

plt.figure(figsize=(10, 10))
plt.title("Loss Graph")
plt.plot(gen_loss_history, label="Generator Loss")
plt.plot(disc_loss_history, label="Discriminator Loss")
plt.ylabel("Loss")
plt.xlabel("Iterators")
plt.legend()
plt.show()

"""#Plot Train Valid Accuracy Curves"""

# Implements plotting of train and validation dataset loss and accuracy curves
def plotTrainValidHistory(train_acc, valid_acc):
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10, 10))


    # Accuracy Curve
    # plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train', color='blue', linestyle='-')
    plt.plot(epochs, valid_acc, label='Validation', color='orange', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    # plt.ylim(0.67, 1)
    # plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.draw()

plotTrainValidHistory(train_acc_history, val_acc_history)

"""#Evaluate Model and visualize predictions"""

# Load our trained discriminator and generator
DISC_LOAD_PATH = ROOT_DIR + '/' + "Task02/new_discriminator_30.pth"
GEN_LOAD_PATH = ROOT_DIR + '/' + "Task02/new_generator_30.pth"

generator = Generator()
discriminator = Discriminator(num_classes=4, dropout_rate=0.2)

trained_discriminator = loadModel(discriminator, DISC_LOAD_PATH)
trained_generator = loadModel(generator, GEN_LOAD_PATH)

test_accuracy, test_loss, all_preds, all_labels = evaluateModel(trained_generator, trained_discriminator, test_loader)

visualizePredictions(test_loader, all_preds, all_labels)

"""#Real Time Input testing"""

DISC_LOAD_PATH = ROOT_DIR + '/' + "Task02/new_discriminator_30.pth"
GEN_LOAD_PATH = ROOT_DIR + '/' + "Task02/new_generator_30.pth"

generator = Generator()
discriminator = Discriminator(num_classes=4, dropout_rate=0.2)
trained_discriminator = loadModel(discriminator, DISC_LOAD_PATH)
trained_generator = loadModel(generator, GEN_LOAD_PATH)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

LATENT_DIM = 100
noise_input = torch.randn(LATENT_DIM, 1, 1, device=device).unsqueeze(dim=0)

predicted_class = predictInputImage(noise_input, trained_generator, trained_discriminator)
print(f'Predicted class: {predicted_class.cpu().numpy()}')

# End of code