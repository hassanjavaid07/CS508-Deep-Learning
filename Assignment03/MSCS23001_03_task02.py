"""
###<u> **DEEP LEARNING PROGRAMMING ASSIGNMENT # 3** </u>
* **NAME = HASSAN JAVAID**
* **ROLL NO. = MSCS23001**
* **TASK01 = Classification of MNIST dataset by custom-built CNN network**
* **TASK02 = Implementation and Transfer Learning of Resnet34**
* **ALGORITHM used: Convolutional Neural Networks (CNNs)**
"""


# ================================================================
# ========================= Task 02 ==============================
# ================================================================


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, models
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import os
import shutil
import pandas as pd
from PIL import Image
import pickle
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import random


# ========================= START OF FUNCTION DEFINITIONS ====================


# Implements the following functions:
# 1. Reading image contents from each subfolder of image_classification folder. 
# 2. Performs random random shuffling of the read image data. 
# 3. Splitting of image data according to given train, val & test split sizes.
# 4. Returns train_files, val_files and test_files for further processing.
def customTrainValTestSplit(files, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=True):
    if shuffle:
        random.shuffle(files)
    total_size = train_size + val_size + test_size
    train_end = int(len(files) * train_size / total_size)
    val_end = train_end + int(len(files) * val_size / total_size)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    return train_files, val_files, test_files



# Implements extraction of class information for reference purposes (optional)
def extractClassInfo(filename, class_names, class_labels):
    # Remove the file extension
    filename = os.path.splitext(filename)[0]
    # Split the filename using delimiters like spaces, dashes, or parentheses
    # parts = filename.split(' ', '-', '(')
    parts = filename.split() + filename.split('-') + filename.split('(')
    for part in parts:
        if part.lower() in class_names:
            class_name = part.lower()
            class_label = class_labels[class_names.index(class_name)]
            return class_name, class_label
    return None, None


# Implements creation of label CSV files from given image folder
def createLabelCSV(folder, csv_filename, class_names, class_labels):
    data = []
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.basename(os.path.join(root, file))
            filename = file_path.split('\\')[-1]
            a = filename.split('_') + filename.split(' ') + filename.split('-') + filename.split('.')
            for ele in a:
                if ele in class_names:
                    class_name = ele
                    class_label = class_labels[class_names.index(ele)]
            # class_name = filename.split('-')[0]
            # class_label = class_labels[class_names.index(class_name)]
            # class_name, class_label = extractClassInfo(filename, class_names, class_labels)
            data.append({'filename': filename, 'class_name': class_name, 'class_label': class_label})
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)



# Implements dataset file saving and loading 
def saveDatasetToFile(filename, dataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def loadDatasetFromFile(filename):
    with open(filename, 'rb') as f:
        loaded_dataset = pickle.load(f)
    return loaded_dataset



# Gets One-Hot Encoding for the given labels
def getOneHotEncoding(encoder, labels):
    return encoder.fit_transform(labels.reshape(-1, 1)).toarray()



# Implements Train-Val split and One-Hot Encoding
def genEncoding(images_tensor, labels_tensor, encoder):
    images_np = images_tensor.numpy()
    labels_np = labels_tensor.numpy()
    # train_images_np, val_images_np, train_labels_np, val_labels_np = getDatasetSplit(images_np, labels_np, split_size=val_size)
    labels_oneHot = getOneHotEncoding(encoder, labels_np)
    # Print the shapes of the one-hot encoded labels
    print("Labels one-hot encoded shape:", labels_oneHot.shape)
    images_tensor = torch.tensor(images_np)
    labels_tensor = torch.tensor(labels_oneHot)

    return TensorDataset(images_tensor, labels_tensor)



# Implements loading of Task-02 datasets from their respective folders
def loadDataset(folderPath, labelFilePath, transform, train=False):
    labels_df = pd.read_csv(labelFilePath)
    tranformed_images = []
    labels = []
    for idx, row in labels_df.iterrows():
        filename = row['filename']
        label = int(row['class_label'])
        img = Image.open(os.path.join(folderPath, filename))
        tranformed_images.append(transform(img))
        labels.append(label)

    images_tensor = torch.stack(tranformed_images)
    labels_tensor = torch.tensor(labels)

    # If it is training or validation dataset then generate One-Hot Encoding and then return them
    if train:
        encoder = OneHotEncoder(categories='auto')
        return genEncoding(images_tensor, labels_tensor, encoder)
    else:
        # Returns test dataset
        return TensorDataset(images_tensor, labels_tensor)


# Implements saving and loading of the model
def loadModel(filename):
    with open(filename, 'rb') as f:
        trained_model = pickle.load(f)
    return trained_model

def saveModel(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Implementation of Resnet Class from scratch. (Task-02, Part-01)
# This implementation is inspired from Kaggle Workbook
# located at: https://www.kaggle.com/code/poonaml/building-resnet34-from-scratch-using-pytorch
# Customized Resnet Block with Resnet34 Compute block implementation
class ResnetBlock(nn.Module):
    def __init__(self, inchannels, outchannels, stride=1, getDownsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.stride = stride
        if getDownsample:
            self.downsample = nn.Sequential(
                    nn.Conv2d(inchannels, outchannels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(outchannels),
                )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity         # skip input
        out = self.relu(out)

        return out

# Implementation of Resnet Class from scratch.
# This implementation is inspired from Kaggle Workbook
# located at: https://www.kaggle.com/code/poonaml/building-resnet34-from-scratch-using-pytorch
class ResNet(nn.Module):

    def __init__(self, block, num_blocks_layer, num_classes=1000):
        super().__init__()

        self.inchannels = 64

        self.conv1 = nn.Conv2d(3, self.inchannels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._gen_Layer(block, 64, num_blocks_layer[0])
        self.layer2 = self._gen_Layer(block, 128, num_blocks_layer[1], stride=2)
        self.layer3 = self._gen_Layer(block, 256, num_blocks_layer[2], stride=2)
        self.layer4 = self._gen_Layer(block, 512, num_blocks_layer[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 , num_classes)


    def _gen_Layer(self, block, outchannels, blocks, stride=1):
        getDownsample = False

        if stride != 1 or self.inchannels != outchannels:
            getDownsample = True

        num_blocks_layer = []
        num_blocks_layer.append(block(self.inchannels, outchannels, stride, getDownsample))

        self.inchannels = outchannels

        for _ in range(1, blocks):
            num_blocks_layer.append(block(self.inchannels, outchannels))

        return nn.Sequential(*num_blocks_layer)


    def forward(self, out):
        out = self.conv1(out)           # 224x224
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)         # 112x112

        out = self.layer1(out)          # 56x56
        out = self.layer2(out)          # 28x28
        out = self.layer3(out)          # 14x14
        out = self.layer4(out)          # 7x7

        out = self.avgpool(out)         # 1x1
        out = torch.flatten(out, 1)     # remove 1 out 1 grid and make vector of tensor shape
        out = self.fc(out)

        return out


# Implementation of Resnet34 model from scratch
def makeResnet34():
    num_blocks_layer=[3, 4, 6, 3]
    model = ResNet(ResnetBlock, num_blocks_layer)
    return model



# Implements training function with early stopping and learning rate decay
def trainModel(model, train_loader, val_loader, criterion, optimizer, num_epochs, max_early_stop, lr_decay_factor):
#     STEPS:
#     1. Check for GPU availability
#     2. Define lists for storing training & validation loss and accuracy history
#     3. Use CrossEntropy as loss function
#     4. Initailize learning rate decay scheduler and early_stopping_counter.
#     5. Iterate over train_loader & val_loader and calculate loss and accuracy for each sample
#     6. If valid_loss is less than best_valid_loss, increase early_stopping_counter.
#     7. Deacrease learning rate by decay factor using scheduler step.
#     8. Store the results in relevant lists
#     9. Finish training after NUM_EPOCHS
#     10. Return Training/Validation loss and accuracy history
    print("Training & validation loop started... ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    best_valid_loss = float('inf')
    early_stopping_counter = 0
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay_factor)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = torch.argmax(labels, dim=1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        print(f"Total Train Samples = {total_samples}")
        train_accuracy = correct_predictions / total_samples
        train_loss /= len(train_loader.dataset)
        train_acc_history.append(train_accuracy)
        train_loss_history.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                labels = torch.argmax(labels, dim=1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                # labels = torch.argmax(labels, dim=1)
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        print(f"Total Validation Samples = {total_samples}")
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_predictions / total_samples
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= max_early_stop:
                print("Early stopping triggered!")
                return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history
                break

        # Learning rate decay
        scheduler.step()

    print("Training finished.")
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history



# Implements function for unfreezing some layers of custom-built Resnet34 model (Task-02, Part-01)
def unfreezeLayers(model, unfreeze_layer='fc'):
    for name, param in model.named_parameters():
        if unfreeze_layer in name:
            param.requires_grad = True
    return model


# Part-01 of Task-02
# Load custom built Resnet34 model from scratch and load pre-determined weights from file
def loadResnet34_part01():
    resnet34 = makeResnet34()
    RESNET_34_WEIGHTS_FILE = "E:\\data\\Univ Data\\ITU MS CS Offline\\DL ML - Datasets\\Resnet34 weights\\resnet34-b627a593.pth"
    weights = torch.load(RESNET_34_WEIGHTS_FILE)
    resnet34.load_state_dict(weights)
    print("Resnet weights successfully loaded from file.")
    # Freeze all the layers except output layer
    for param in resnet34.parameters():
        param.requires_grad = False
    
    # Unfreeze selected layers
    # resnet34 = unfreezeLayers(resnet34, 'fc')
    resnet34 = unfreezeLayers(resnet34, 'layer4')
    resnet34 = unfreezeLayers(resnet34, 'layer3')
    # resnet34 = unfreezeLayers(resnet34, 'layer2')
    # resnet34 = unfreezeLayers(resnet34, 'layer1')
    
    # Substitute the customized FC output layer and initialize it
    resnet34.fc = torch.nn.Linear(resnet34.fc.in_features, len(class_names))
    resnet34.fc.requires_grad_ = True
    torch.nn.init.xavier_uniform_(resnet34.fc.weight)
    
    for name, param in resnet34.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
    
    return resnet34


# Part-02 of Task-02
# Load Pre-Trained Resnet34 Model
def loadResnet34_part02():
    resnet34 = models.resnet34(pretrained=True)
    # Substitute the customized FC output layer and initialize it
    resnet34.fc = torch.nn.Linear(resnet34.fc.in_features, 7)
    torch.nn.init.xavier_uniform_(resnet34.fc.weight)
    return resnet34


# Implements the test function to evaluate our trained model
def evaluateModel(model, test_loader, num_epochs):
    model.eval()
    test_history = {'accuracy': [], 'f1-score': []}
    for epoch in range(num_epochs):
      with torch.no_grad():
          all_preds = []
          all_labels = []
          for images, labels in test_loader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              _, preds = torch.max(outputs, 1)
              all_preds.extend(preds.tolist())
              all_labels.extend(labels.tolist())

          conf_matrix = confusion_matrix(all_labels, all_preds)
          f1 = f1_score(all_labels, all_preds, average='macro')
          accuracy = accuracy_score(all_labels, all_preds)
      print(f"Epoch {epoch+1}/{num_epochs} - Test Accuracy: {accuracy:.4f} - F1-Score: {f1:.4f}")
      test_history['accuracy'].append(accuracy)
      test_history['f1-score'].append(f1)

    return test_history, conf_matrix, all_preds, all_labels


# ==========================================================================================
# Implementation trained_model plotting and visualization functions on train/valid/test datasets

# Implements plotting of train and validation dataset loss and accuracy curves
def plotTrainValidHistory(train_loss, valid_loss, train_acc, valid_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train', color='blue', linestyle='-')
    plt.plot(epochs, valid_loss, label='Validation', color='orange', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.ylim(0.05, 2.85)
    plt.grid(True)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train', color='blue', linestyle='-')
    plt.plot(epochs, valid_acc, label='Validation', color='orange', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.draw()


# Implements plotting of confusion matrix for test dataset
def plotConfusionMatrix(confusion_matrix):
    classes = class_names
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.draw()



# Implements plotting of sample images from dataset
def plotSampleImages(dataset, suptitle, oneHot=False):
    fig = plt.figure(figsize=(12, 12))
    rows, cols = 4, 4
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(dataset), size=[1]).item()
        img, label = dataset[random_idx]
        if oneHot:
            label = np.argmax(label)
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(int(label), fontsize=12, fontweight='bold')
        ax.axis('off')
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.draw()



# Implements visualization of correct and wrong predictions
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
    for i in indices:
        image, _ = test_loader.dataset[i]
        images.append(image)
    preds = [all_preds[i] for i in indices]
    images = torch.stack(images)
    plt.figure(figsize=(10, 5))
    plt.suptitle(title)
    for i in range(len(indices)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f'Predicted Label: {class_names[preds[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.draw()


# Implements the plotting of test accuracy, f1-score curves
def plotTestHistory(test_history, epochs):
    fig = plt.figure(figsize=(12, 6))

    plt.plot(np.arange(epochs), test_history['accuracy'], label='Accuracy', color='red')
    plt.plot(np.arange(epochs), test_history['f1-score'], label='F1-Score', color='brown')

    plt.title('Test Accuracy & f1-score curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy / F1-Score')
    plt.xticks(np.arange(0, epochs+1, 10))
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    final_test_accuracy = test_history['accuracy'][-1]
    final_f1_score = test_history['f1-score'][-1]
    fig.suptitle(f'''Final Test Accuracy: {final_test_accuracy:.4f}, Final F1-Score: {final_f1_score:.4f}''', fontsize=14)
    plt.tight_layout()
    plt.show()


# Implements functionlity for real-time image preprocessing and class prediction
def preprocessImage(test_image_path, transform):
    img = Image.open(test_image_path)

    
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return img_tensor


def predictInputImage(test_image_path, model, transform):
    # Preprocess the image
    img_tensor = preprocessImage(test_image_path, transform)

    model.eval()

    # Perform prediction
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted


# ======================= END OF FUNCTION DEFINTIONS ====================================

# Change ROOT_DIR & DESTINATION_DIR to your desired path names
ROOT_DIR = 'E:\\data\\Univ Data\\ITU MS CS Offline\\DL ML - Datasets'
DESTINATION_DIR = 'E:\\data\\Univ Data\\ITU MS CS Offline\\DL ML - Datasets\\Processed_task02'
# Replace 'TEST_IMAGE_PATH' with the path to the image you want to predict
TEST_IMAGE_PATH = ROOT_DIR + "\\" + "Test_Images\\7.jpeg"

# Defines the source folder for image classification dataset folder. Change this to your folder path
sourceFolder = ROOT_DIR + "\\image_classification"

# Defines the destination folder for train, valid & test dataset folder
destinationFolder = DESTINATION_DIR


print("\n")
print("==================================================================================")
print("==================================================================================")
print(f"Reading image data from root directory: {ROOT_DIR}")
# Make destination folder if it does not exist
if not os.path.exists(destinationFolder):
    os.makedirs(destinationFolder)

# Copy files to destinatin folder
for category in os.listdir(sourceFolder):
    category_path = os.path.join(sourceFolder, category)
    if os.path.isdir(category_path):
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            shutil.copy(file_path, destinationFolder)

all_image_files = os.listdir(destinationFolder)

print(f"All Images Loaded. Number of total images = {len(all_image_files)}")
print("==================================================================================")
print(f"Splitting all image files into train, valid and test image files...")
# Split the image classification dataset files to train, valid and test image files
train_files, val_files, test_files = customTrainValTestSplit(all_image_files)

# Define train, validation and test folder paths
train_folder = destinationFolder + "\\" + "train_folder"
val_folder = destinationFolder + "\\" + "validation_folder"
test_folder = destinationFolder + "\\" + "test_folder"
print("==================================================================================")

print("\n")
print("==================================================================================")
print("Moving image files into relevant folders...")
# Create destination folders if they already not exist
for folder in [train_folder, val_folder, test_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

for file in train_files:
    src = os.path.join(destinationFolder, file)
    dst = os.path.join(train_folder, file)
    shutil.move(src, dst)

for file in val_files:
    src = os.path.join(destinationFolder, file)
    dst = os.path.join(val_folder, file)
    shutil.move(src, dst)

for file in test_files:
    src = os.path.join(destinationFolder, file)
    dst = os.path.join(test_folder, file)
    shutil.move(src, dst)

print(f"Train folder created at: {train_folder}")
print(f"Validation folder created at: {val_folder}")
print(f"Test folder created at: {test_folder}")

# Define class label information
class_names = ['bike', 'car', 'cat', 'dog', 'flower', 'horse', 'rider']
class_labels = [0, 1, 2, 3, 4, 5, 6]
class_dict = dict(zip(class_names, class_labels))

# Loaded from ImageNet Paper resource
# https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
paramsImageNet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
mean = paramsImageNet[0]
std = paramsImageNet[1]


# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# Define train, validation and test label csv file paths
TRAIN_LABEL_CSV_FILE = destinationFolder + "\\" + "train_labels.csv"
VAL_LABEL_CSV_FILE = destinationFolder + "\\" + "val_labels.csv"
TEST_LABEL_CSV_FILE = destinationFolder + "\\" + "test_labels.csv"

print("\n")
print("==================================================================================")
print("Creating label csv files....")
# Create label CSV files
createLabelCSV(train_folder, TRAIN_LABEL_CSV_FILE, class_names, class_labels)
createLabelCSV(val_folder, VAL_LABEL_CSV_FILE, class_names, class_labels)
createLabelCSV(test_folder, TEST_LABEL_CSV_FILE, class_names, class_labels)

print(f"Train label file created at: {TRAIN_LABEL_CSV_FILE}")
print(f"Validation label file created at: {VAL_LABEL_CSV_FILE}")
print(f"Test label file created at: {TEST_LABEL_CSV_FILE}")


# Load train, validation and test dataset from respective folders
train_dataset = loadDataset(train_folder, TRAIN_LABEL_CSV_FILE, transform, train=True)
val_dataset = loadDataset(val_folder, VAL_LABEL_CSV_FILE, transform, train=True)
test_dataset = loadDataset(test_folder, TEST_LABEL_CSV_FILE, transform, train=False)



# =======================================================================================
# # # Save the datasets for future uses in training to avoid reloading dataset every time
# # Dataset save file names
TRAIN_DATASET_SAVE_NAME = destinationFolder + "\\" + 'train_dataset_task02.pkl'
VAL_DATASET_SAVE_NAME = destinationFolder + "\\" + 'val_dataset_task02.pkl'
TEST_DATASET_SAVE_NAME = destinationFolder + "\\" + 'test_dataset_task02.pkl'

print("\n")
print("==================================================================================")
print("Saving all datasets for future uses in training to avoid reloading...")
saveDatasetToFile(TRAIN_DATASET_SAVE_NAME, train_dataset)
print("Train dataset saved to:", os.path.abspath(TRAIN_DATASET_SAVE_NAME))

saveDatasetToFile(VAL_DATASET_SAVE_NAME, val_dataset)
print("Validation dataset saved to:", os.path.abspath(VAL_DATASET_SAVE_NAME))

saveDatasetToFile(TEST_DATASET_SAVE_NAME, test_dataset)
print("Test dataset saved to:", os.path.abspath(TEST_DATASET_SAVE_NAME))


train_dataset = loadDatasetFromFile(TRAIN_DATASET_SAVE_NAME)
print("Train dataset loaded from:", os.path.abspath(TRAIN_DATASET_SAVE_NAME))
val_dataset = loadDatasetFromFile(VAL_DATASET_SAVE_NAME)
print("Validation dataset loaded from:", os.path.abspath(VAL_DATASET_SAVE_NAME))
test_dataset = loadDatasetFromFile(TEST_DATASET_SAVE_NAME)
print("Test dataset loaded from:", os.path.abspath(TEST_DATASET_SAVE_NAME))
# ========================================================================================

print("\n")
print("==================================================================================")
# Print the lengths of the datasets
print(f"Training dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")


# Define hyper parameters
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.00001
EARLY_STOP_FACTOR = 7
LR_DECAY_FACTOR = 0.0005

# Prepare Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# For doing Task-02 Part-01, use this definition
# model = loadResnet34_part01()


# For doing Task-02 Part-02, use this definition
model = loadResnet34_part02()


print("\n")
print("==================================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device = {device}")

# Print Summary of our Resnet34 model
summary(model.to(device), (3, 224, 224))



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)# Define training parameters

print("\n")
print("==================================================================================")
print("Training started...")
# Train the model with early stopping and learning rate decay
trained_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = trainModel(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOP_FACTOR, LR_DECAY_FACTOR)


# #================= UNCOMMENT ONLY IF YOU WANT TO TEST SAVE FUNCTION =================
SAVE_MODEL_FILE_NAME = destinationFolder + "\\" + "trained_modeltask02part01_bs0" + str(BATCH_SIZE) + ".pkl"
saveModel(trained_model, SAVE_MODEL_FILE_NAME)


# Evaluate the model
test_history, conf_matrix, all_preds, all_labels = evaluateModel(trained_model, test_loader, NUM_EPOCHS)
f1 = test_history['accuracy'][-1]
accuracy = test_history['accuracy'][-1]
print(f'Test Data F1 Score: {f1:.4f}')
print(f'Test Data Accuracy: {accuracy:.4f}')



# Visualize the results
plotTrainValidHistory(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

plotConfusionMatrix(conf_matrix)

visualizePredictions(test_loader, all_preds, all_labels)

plotTestHistory(test_history, NUM_EPOCHS)


# Prediction of real-time image inputs
LOAD_MODEL_FILE_NAME = SAVE_MODEL_FILE_NAME
loaded_model = loadModel(LOAD_MODEL_FILE_NAME)
predicted_class = predictInputImage(TEST_IMAGE_PATH, loaded_model, transform)
print(f'Predicted class: {predicted_class.cpu().numpy()}')

plt.show()

#End of Code