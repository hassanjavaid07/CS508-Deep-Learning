### DEEP LEARNING PROGRAMMING ASSIGNMENT # 1
# NAME = HASSAN JAVAID
# ROLL NO. = MSCS23001
# TASK = BINARY CLASSIFICATION THRU LOGISTIC REGRESSION
# OPTIM ALGORITHM: STOCHASTIC GRADIENT DESCENT (SGD)

# LOGISTIC REGRESSION (TASK-02)
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


label_encoder = LabelEncoder() # Initialize LabelEncoder


def loadAndPreprocessData(filePath, label_encoder=label_encoder, train=True):
    df = pd.read_csv(filePath)
    if train:
        X = df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
        y = df['Survived']
    else:
        X = df.drop(columns=['PassengerId', 'Name', 'Ticket'])
        y = None
    
    # Perform label encoding on Embarked, Sex columns.
    X_preprocessed = X.copy()
    col_name = 'Embarked'
    X_preprocessed_en = X_preprocessed[col_name]
    encoded_column = label_encoder.fit_transform(X_preprocessed_en)
    X_preprocessed[col_name] = encoded_column

    col_name = 'Sex'
    X_preprocessed_en = X_preprocessed[col_name]
    encoded_column = label_encoder.fit_transform(X_preprocessed_en)
    X_preprocessed[col_name] = encoded_column

    # Fill in blank values in columns
    col_name = 'Cabin'
    # Replace NaN values with the most frequent value
    most_frequent_value = X_preprocessed[col_name].mode()[0]
    X_preprocessed.fillna({col_name: most_frequent_value}, inplace=True)
    # Apply label encoding to the column
    X_preprocessed_en = X_preprocessed[col_name]
    X_preprocessed[col_name] = label_encoder.fit_transform(X_preprocessed_en)

    col_name = 'Age'
    # Replace NaN values with the most frequent value
    most_frequent_value = X_preprocessed[col_name].mode()[0]
    X_preprocessed.fillna({col_name: most_frequent_value}, inplace=True)
    # Apply label encoding to the column
    X_preprocessed_en = X_preprocessed[col_name]
    X_preprocessed[col_name] = label_encoder.fit_transform(X_preprocessed_en)

    col_name = 'Fare'
    # Replace NaN values with the most frequent value
    most_frequent_value = X_preprocessed[col_name].mode()[0]
    X_preprocessed.fillna({col_name: most_frequent_value}, inplace=True)
    # Apply label encoding to the column
    X_preprocessed_en = X_preprocessed[col_name]
    X_preprocessed[col_name] = label_encoder.fit_transform(X_preprocessed_en)

    return X_preprocessed, y


# Split the input_data into datasets as per use
def dataSplit(X, y, X1_size=0.9, X2_size=0.1, SHUFFLE=False):
    """
    Splits the input data as per use. Output is numpy arrays.
    """
    X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=X2_size, shuffle=SHUFFLE)
    return X_1.values, y_1.values, X_2.values, y_2.values


def normalize(X, mean, variance):
    """
    Normalize the columns of X.
    """
    X_normalized = (X - mean) / np.sqrt(variance)
    return X_normalized


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Apply the forward eqn: y_hat = X.theta
def feedForward(X, theta):
    z = np.dot(X, theta)
    a = sigmoid(z)
    return a

# Compute binary cross-entropy loss
def binaryLoss(y, y_hat):
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


# Compute the gradient
def computGradient(X, y, y_hat):
    return np.mean(np.dot(X.T, (y_hat - y)))


# Apply Stochastic Gradient Descent (SGD) optimSGD
def optimSGD(lr, grad, theta):
    theta -= lr * grad
    return theta


# Initialize the logistic Regression Model
def logisticRegressionModel(N=8, seed=None):
    #Model_1: np.random.randn(N)
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(N)
    
    #Model_2: np.ones(N)
    # return np.ones(N)
    
    #Model_3: np.zeros(N)
    # return np.zeros(N)



# Plot the training and validation history
def plotHistory(train_history, val_history, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE):
    # Create a figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Plot training and validation loss history
    axes[0].plot(np.arange(NUM_EPOCHS), train_history['loss'], label='Training Loss', color='blue')
    axes[0].plot(np.arange(NUM_EPOCHS), val_history['loss'], label='Validation Loss', color='red')
    axes[0].set_title('Training Loss vs Validation Loss', fontsize=12)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_xticks(np.arange(0, NUM_EPOCHS+1, 10))
    axes[0].legend()
    axes[0].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Plot training and validation accuracy history
    axes[1].plot(np.arange(NUM_EPOCHS), train_history['accuracy'], label='Training Accuracy', color='blue')
    axes[1].plot(np.arange(NUM_EPOCHS), val_history['accuracy'], label='Validation Accuracy', color='red')
    axes[1].set_title('Training Accuracy vs Validation Accuracy', fontsize=16)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xticks(np.arange(0, NUM_EPOCHS+1, 10))
    axes[1].legend()
    axes[1].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    final_train_loss = train_history['loss'][-1]
    final_train_accuracy = train_history['accuracy'][-1]
    final_val_loss = val_history['loss'][-1]
    final_val_accuracy = val_history['accuracy'][-1]
    # Adjust layout
    fig.suptitle(f'DL-Asm01-Task02-Logistic Regression\nLearning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}\nFinal Train Loss: {final_train_loss:.4f}, Final Train Accuracy: {final_train_accuracy:.4f}\nFinal Val Loss: {final_val_loss:.4f}, Final Val Accuracy: {final_val_accuracy:.4f}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show the plot
    plt.show()


# Plot testing history
def plotTestHistory(test_history, y_true, y_pred, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE):
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(np.arange(NUM_EPOCHS), test_history['loss'], label='Testing Loss', color='blue')
    plt.plot(np.arange(NUM_EPOCHS), test_history['accuracy'], label='Testing Accuracy', color='red')
    plt.plot(np.arange(NUM_EPOCHS), test_history['f1-score'], label='F1-Score', color='brown')

    plt.title('Testing Loss vs Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy / F1-Score')
    plt.xticks(np.arange(0, NUM_EPOCHS+1, 10))
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    final_test_loss = test_history['loss'][-1]
    final_test_accuracy = test_history['accuracy'][-1]
    final_f1_score = test_history['f1-score'][-1]
    fig.suptitle(f'''DL-Asm01-Task02-Logistic Regression\nLearning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}\nFinal Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_accuracy:.4f}, Final F1-Score: {final_f1_score:.4f}''', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.title(f'''DL-Asm01-Task02-Logistic Regression\nConfusion Matrix\nLearning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}\nFinal Test Accuracy: {final_test_accuracy:.4f}, Final F1-Score: {final_f1_score:.4f}''', fontsize=16)
    plt.tight_layout()
    plt.show()


    
# Define the Logistic Regression implemented using Stochastic Gradient Descent with given lr, batch size
def train(X_train, y_train, X_val, y_val, model, lr, n_epochs, batch_size):
    m, n = X_train.shape
    theta = model
    # print(theta)
    train_losses = []
    train_accuracies = []
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}

    for epoch in range(n_epochs):
        for i in range(0, m, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_hat_batch = feedForward(X_batch, theta)
            gradients = computGradient(X_batch, y_batch, y_hat_batch)
            theta = optimSGD(lr, gradients, theta)
            train_losses.append(binaryLoss(y_batch, y_hat_batch))
            train_accuracies.append(accuracy_score(y_batch, predict(X_batch, theta)))
            # print(theta.shape)

        train_loss = np.mean(train_losses)
        train_accuracy = np.mean(train_accuracies)
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_accuracy)

        y_hat_val = feedForward(X_val, theta)
        val_loss = binaryLoss(y_val, y_hat_val)
        val_accuracy = accuracy_score(y_val, predict(X_val, theta))
        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")


    return theta, train_history, val_history


# Predict the labels for the given features using the trained model.
def predict(X, theta):
    """
    Predict the labels for the given features using the trained model.
    """
    y_pred = feedForward(X, theta)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    return y_pred


def testModel(model, X_test, y_test, n_epochs):
    theta = model
    test_history = {'loss': [], 'accuracy': [], 'f1-score': []}
    for epoch in range(n_epochs):
        test_loss = binaryLoss(y_test, feedForward(X_test, theta))
        test_history['loss'].append(test_loss)
        y_pred = predict(X_test, theta)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_history['accuracy'].append(test_accuracy)
        f1 = f1_score(y_test, y_pred)
        test_history['f1-score'].append(f1)
        print(f"Epoch {epoch+1}/{n_epochs} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f} - F1-Score: {f1:.4f}")
    return test_history, y_pred


def saveModel(theta_final, train_mean, train_variance, filename):
    with open(filename, 'wb') as f:
        pickle.dump((theta_final, train_mean, train_variance), f)


def loadModel(filename):
    with open(filename, 'rb') as f:
        theta_final, train_mean, train_variance = pickle.load(f)
    return theta_final, train_mean, train_variance


# Define root directories and file paths
ROOT_DIR = 'C:\\Users\\HP\\OneDrive\\Documents\\Univ Data\\ITU CS\\Deep Learning\\Assignments\\Assignment01\\Submitted\\Hassan_MSCS23001_01\\'
train_file_name = 'train.csv'
test_file_name = 'test.csv'
test_label_file = 'test_labels.csv'

TRAIN_FILE_PATH = ROOT_DIR + train_file_name
TEST_FILE_PATH = ROOT_DIR + test_file_name
TEST_LABEL_FILE_PATH = ROOT_DIR + test_label_file


# Load and preprocess the dataset
X_preprocessed, y = loadAndPreprocessData(TRAIN_FILE_PATH, label_encoder=label_encoder)


# Split and normalize the data into training and validation dataset
train_size = 0.9
val_size = 0.1
X_train, y_train, X_val, y_val = dataSplit(X_preprocessed, y, X1_size=train_size, X2_size=val_size, SHUFFLE=False)


# Calculate the mean and variance of the training data
train_mean = X_train.mean(axis=0)
train_variance = X_train.var(axis=0)


# Normalize the train and validation datasets
X_train_normalized = normalize(X_train, train_mean, train_variance)
X_val_normalized = normalize(X_val, train_mean, train_variance)


# Define hyperparameters
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
BATCH_SIZE = 8


# Initialize the Logistic Regression Model
model = logisticRegressionModel(seed=121)


# Train the model using stochastic gradient descent with batch size
print("\n\n============================ TRAINING =========================================")
theta_final, train_history, val_history = train(X_train_normalized, y_train, X_val_normalized, y_val, model, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE)



# Save the final model
model_save_name = ROOT_DIR + 'task02.pkl'
saveModel(theta_final, train_mean, train_variance, model_save_name)
print(f'\nModel saved to drive: {model_save_name}')
print()


# Load the model for testing
loaded_model, train_mean, train_variance = loadModel(model_save_name)


# Load and preprocess the test datset
X_test = loadAndPreprocessData(TEST_FILE_PATH, label_encoder=label_encoder, train=False)[0].values
y_test_df = pd.read_csv(TEST_LABEL_FILE_PATH)
y_test = y_test_df['Survived'].values
# print(X_test.shape)
# print(y_test.shape)
# print(type(X_test))
# print(type(y_test))
# print(train_mean)
# print(type(train_mean))
# print(train_variance)
# print(type(train_variance))


# np.savetxt(ROOT_DIR+'out.csv', X_test, delimiter=',')
    
# Normalize the test dataset
X_test_normalized = normalize(X_test, train_mean, train_variance)


# Test the model
print("\n\n============================ TESTING =========================================")
test_history, y_test_pred = testModel(loaded_model, X_test_normalized, y_test, NUM_EPOCHS)



# Plot the training and validation history
plotHistory(train_history, val_history, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE)

# Plot test history and confusion matrix
plotTestHistory(test_history, y_test, y_test_pred, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE)

