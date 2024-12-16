"""
###<u> **DEEP LEARNING PROGRAMMING ASSIGNMENT # 3** </u>
* **NAME = HASSAN JAVAID**
* **ROLL NO. = MSCS23001**
* **TASK01 = Classification of MNIST dataset by custom-built CNN network**
* **TASK02 = Implementation and Transfer Learning of Resnet34**
* **ALGORITHM used: Convolutional Neural Networks (CNNs)**
"""

# ================================================================
# ========================= Task 01 ==============================
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import os
import pandas as pd
from PIL import Image
import pickle
import seaborn as sns
import random
from PIL import Image



# ========================= START OF FUNCTION DEFINITIONS ====================

# Implements dataset file saving and loading 
def saveDatasetToFile(filename, dataset):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)


def loadDatasetFromFile(filename):
    with open(filename, 'rb') as f:
        loaded_dataset = pickle.load(f)
    return loaded_dataset



# Implements the random shuffling of dataset and then splits it
def getDatasetSplit(data, labels, split_size, random_seed=87):

    num_samples = data.shape[0]
    num_split_samples = int(split_size * num_samples)

    indices = np.random.permutation(num_samples)
    shuffled_data = data[indices]
    shuffled_labels = labels[indices]

    data_1 = shuffled_data[num_split_samples:]
    data_2 = shuffled_data[:num_split_samples]
    data_1_labels = shuffled_labels[num_split_samples:]
    data_2_labels = shuffled_labels[:num_split_samples]

    return data_1, data_2, data_1_labels, data_2_labels



# Gets One-Hot Encoding for the given labels
def getOneHotEncoding(encoder, labels):
    return encoder.fit_transform(labels.reshape(-1, 1)).toarray()


# Implements Train-Val split and One-Hot Encoding
def genSplitAndEncoding(images_tensor, labels_tensor, encoder, val_size=0.1):
    images_np = images_tensor.numpy()
    labels_np = labels_tensor.numpy()
    train_images_np, val_images_np, train_labels_np, val_labels_np = getDatasetSplit(images_np, labels_np, split_size=val_size)
    train_labels_oneHot = getOneHotEncoding(encoder, train_labels_np)
    val_labels_oneHot = getOneHotEncoding(encoder, val_labels_np)
    # Print the shapes of the one-hot encoded labels
    print("Train labels one-hot encoded shape:", train_labels_oneHot.shape)
    print("Validation labels one-hot encoded shape:", val_labels_oneHot.shape)
    train_images_tensor = torch.tensor(train_images_np)
    val_images_tensor = torch.tensor(val_images_np)
    train_labels_tensor = torch.tensor(train_labels_oneHot)
    val_labels_tensor = torch.tensor(val_labels_oneHot)

    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)

    return (train_dataset, val_dataset)


# Implements loading of Task-01 datasets from their respective folders
def loadDataset(folderPath, labelFilePath, transform, train=False):
    labels_df = pd.read_csv(labelFilePath)
    tranformed_images = []
    labels = []
    for idx, row in labels_df.iterrows():
        # print('row = ', row)
        # print('idx = ', idx)
        filename = row['Filename']
        label = int(row['Label'])
        img = Image.open(os.path.join(folderPath, filename))
        tranformed_images.append(transform(img))
        labels.append(label)

    images_tensor = torch.stack(tranformed_images)
    labels_tensor = torch.tensor(labels)

    # If it is training dataset then split into train and validation datasets and then return them
    if train:
        encoder = OneHotEncoder(categories='auto')
        return genSplitAndEncoding(images_tensor, labels_tensor, encoder, val_size=0.1)
    else:
        # Returns test dataset
        return TensorDataset(images_tensor, labels_tensor)



# Design CNN network for MNIST classification
# Implements Network-01 for Task-01 (without dropout layers)
class HW03CNN_01_Network01(nn.Module):
    def __init__(self, num_classes):
        super(HW03CNN_01_Network01, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) #14x14


        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) #7x7

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x



# Implements Network-02 for Task-01 (with dropout layers)
class HW03CNN_01_Network02(nn.Module):
    def __init__(self, num_classes, dropout_rate):
        super(HW03CNN_01_Network02, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()



    def forward(self, x):
        x = self.maxpool1(self.dropout1(self.relu(self.bn1(self.conv1(x)))))
        x = self.maxpool2(self.dropout2(self.relu(self.bn2(self.conv2(x)))))
        x = self.flatten(x)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x



# Implements saving and loading of the model
def loadModel(filename):
    with open(filename, 'rb') as f:
        trained_model = pickle.load(f)
    return trained_model



def saveModel(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)



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
            idx, predicted = torch.max(outputs, 1)
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
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                labels = torch.argmax(labels, dim=1)
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



# Implements the test function to evaluate our trained model
def evaluateModel(model, test_loader, num_epochs):
    model.eval()
    classes = list(range(10))
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
# Implements trained_model plotting and visualization functions on train/valid/test datasets
# ==========================================================================================

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
    plt.ylim(0.05, 1.05)
    plt.grid(True)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train', color='blue', linestyle='-')
    plt.plot(epochs, valid_acc, label='Validation', color='orange', linestyle='-')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.ylim(0.67, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.draw()


# Implements plotting of confusion matrix for test dataset
def plotConfusionMatrix(confusion_matrix):
    classes = list(range(10))
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
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
        plt.title(f'Predicted Label: {preds[i]}')
        plt.axis('off')
    plt.tight_layout()
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
    plt.draw()


# Implements functionlity for real-time image preprocessing and class prediction
def preprocessImage(test_image_path):
    img = Image.open(test_image_path)

    # Define transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),                  # Resize to 28x28
        transforms.ToTensor(),                        # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))          # Normalize the pixel values
    ])

    # Apply transformations to the image
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return img_tensor

def predictInputImage(test_image_path, model):
    # Preprocess the image
    img_tensor = preprocessImage(test_image_path)

    model.eval()

    # Perform prediction
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted

# ======================= END OF FUNCTION DEFINTIONS ====================================


# Change ROOT_DIR & DESTINATION_DIR to your desired path name
ROOT_DIR = "E:\\data\\Univ Data\\ITU MS CS Offline\\DL ML - Datasets"
DESTINATION_DIR = "E:\\data\\Univ Data\\ITU MS CS Offline\\DL ML - Datasets\\Processed_task01"
# Replace 'TEST_IMAGE_PATH' with the path to the image you want to predict
TEST_IMAGE_PATH = ROOT_DIR + "\\" + "Test_Images\\7.jpeg"

# Defines the destination folder for train, valid & test dataset save and load
destinationFolder = DESTINATION_DIR



# Define folder paths
train_folder = ROOT_DIR + "\\DLA3_MNIST_Data" + "\\" + "dla3_train\\train"
test_folder = ROOT_DIR + "\\DLA3_MNIST_Data" + "\\" + "dla3_test\\test"
train_label_file = ROOT_DIR + "\\DLA3_MNIST_Data" + "\\" + "dla3_train\\train.csv"
test_label_file = ROOT_DIR + "\\DLA3_MNIST_Data" + "\\" + "dla3_test\\test.csv"

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ==================================================================================================================
# # # Save the datasets for future uses in training to avoid reloading dataset every time
# # Dataset save file names
TRAIN_DATASET_SAVE_NAME = destinationFolder + "\\" + "train_dataset_task01.pkl"
VAL_DATASET_SAVE_NAME = destinationFolder + "\\" + "val_dataset_task01.pkl"
TEST_DATASET_SAVE_NAME = destinationFolder + "\\" + "test_dataset_task01.pkl"

print("\n")
print("==================================================================================")
print(f"Loading datasets from folders in Root Directory: {ROOT_DIR}")
# ===================================================================================================================
# ======== RUN THIS CODE ONLY ONCE FOR LOADING DATASET FROM FILE AND SAVING THEM TO LOCAL DISK ======================
# ===================================================================================================================
# Load and create datasets and save them to file for future training use
# Perform One-Hot Encoding
encoder = OneHotEncoder(categories='auto')
dataset_pre = loadDataset(train_folder, train_label_file, transform=transform,  train=True)
train_dataset, val_dataset = dataset_pre[0], dataset_pre[1]
test_dataset_pre = loadDataset(test_folder, test_label_file, transform=transform, train=False)
print(f"Train Dataset loaded from {train_folder}")
print(f"Validation Dataset created from {train_folder}")
print(f"Test Dataset loaded from {test_folder}")
print()

# Save the datasets for future uses in training to avoid reloading
saveDatasetToFile(TRAIN_DATASET_SAVE_NAME, train_dataset)
print("Train dataset saved to:", os.path.abspath(TRAIN_DATASET_SAVE_NAME))

saveDatasetToFile(VAL_DATASET_SAVE_NAME, val_dataset)
print("Validation dataset saved to:", os.path.abspath(VAL_DATASET_SAVE_NAME))

saveDatasetToFile(TEST_DATASET_SAVE_NAME, test_dataset_pre)
print("Test dataset saved to:", os.path.abspath(TEST_DATASET_SAVE_NAME))
print()
# ==================================================================================================================
train_dataset = loadDatasetFromFile(TRAIN_DATASET_SAVE_NAME)
print("Train dataset loaded from:", os.path.abspath(TRAIN_DATASET_SAVE_NAME))
print("Sample Train images loaded...")
plotSampleImages(train_dataset, "Sample Train Images", oneHot=True)
print()

val_dataset = loadDatasetFromFile(VAL_DATASET_SAVE_NAME)
print("Validation dataset loaded from:", os.path.abspath(VAL_DATASET_SAVE_NAME))
print("Sample Validation images loaded...")
plotSampleImages(val_dataset, "Sample Validation Images", oneHot=True)
print()

test_dataset = loadDatasetFromFile(TEST_DATASET_SAVE_NAME)
print("Test dataset loaded from:", os.path.abspath(TEST_DATASET_SAVE_NAME))
print("Sample Test images loaded...")
plotSampleImages(test_dataset, "Sample Test Images")
print()
# ===================================================================================================================
# ======================================== END OF LOAD AND SAVE CODE ================================================
# ===================================================================================================================

print("\n")
print("==================================================================================")
# Print the lengths of the datasets
print(f"Training dataset length: {len(train_dataset)}")
print(f"Validation dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")


# Define hyper parameters
BATCH_SIZE = 128
NUM_EPOCHS = 25
EARLY_STOP_FACTOR = 10
DROPOUT_RATE = 0.5
LR_DECAY_FACTOR = 0.005
LEARNING_RATE = 0.1



# Initialize the network models (network-01 & network-02)
num_classes = 10
# For network-01 use this definition
model = HW03CNN_01_Network01(num_classes)

# For network-02 use this definition
# model = HW03CNN_01_Network02(num_classes, DROPOUT_RATE)

print("\n")
print("==================================================================================")
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device = {device}")

# Display the model architecture
summary(model.to(device), (1, 28, 28))


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Prepare Data Loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Train the model with early stopping and learning rate decay
trained_model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = trainModel(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, EARLY_STOP_FACTOR, LR_DECAY_FACTOR)


#================= UNCOMMENT ONLY IF YOU WANT TO TEST SAVE FUNCTION =================
SAVE_MODEL_FILE_NAME = destinationFolder + "\\" + "trained_modeltask01network01.pkl"
saveModel(trained_model, SAVE_MODEL_FILE_NAME)


# Evaluate the model and display accuracy and f1-score
test_history, conf_matrix, all_preds, all_labels = evaluateModel(trained_model, test_loader, NUM_EPOCHS)
f1 = test_history['accuracy'][-1]
accuracy = test_history['accuracy'][-1]
print(f'Test Data F1 Score: {f1:.4f}')
print(f'Test Data Accuracy: {accuracy:.4f}')


# Visualize results
plotTrainValidHistory(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

plotConfusionMatrix(conf_matrix)

visualizePredictions(test_loader, all_preds, all_labels)

plotTestHistory(test_history, NUM_EPOCHS)


# Prediction of real-time image inputs


LOAD_MODEL_FILE_NAME = SAVE_MODEL_FILE_NAME
loaded_model = loadModel(LOAD_MODEL_FILE_NAME)
predicted_class = predictInputImage(TEST_IMAGE_PATH, loaded_model)
print(f'Predicted class: {predicted_class.cpu().numpy()}')

plt.show()
#End of Code