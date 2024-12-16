### DEEP LEARNING PROGRAMMING ASSIGNMENT # 1
# NAME = HASSAN JAVAID
# ROLL NO. = MSCS23001
# TASK = LINEAR REGRESSION
# OPTIM ALGORITHM: STOCHASTIC GRADIENT DESCENT (SGD)

#Linear Regression (TASK-01)
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


# Apply the forward eqn: y_hat = X.theta
def feedForward(X, theta):
    return np.dot(X, theta)


# Compute l2 loss
def l2Loss(y, y_hat):
    return (1/2) * np.mean(y_hat - y)**2


# Compute the gradient
def computeGradient(X, y, y_hat):
    return np.mean(np.dot(X.T, (y_hat - y)))


# Apply Stochastic Gradient Descent (SGD) optimSGD
def optimSGD(lr, grad, theta):
    theta -= lr*grad
    return theta


# Initialize the linear Regression Model
def linearRegressionModel(N=8, seed=None):
    #Model_1: np.random.randn(N)
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(N)
    
    #Model_2: np.ones(N)
    # return np.ones(N)
    
    #Model_3: np.zeros(N)
    # return np.zeros(N)


# Plot training and validation loss
def plotHistory(train_history, val_history, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE):
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(np.arange(NUM_EPOCHS), train_history['loss'], label='Training Loss', color='blue')
    plt.plot(np.arange(NUM_EPOCHS), val_history['loss'], label='Validation Loss', color='red')

    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, NUM_EPOCHS+1, 10))
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    final_train_loss = train_history['loss'][-1]
    final_val_loss = val_history['loss'][-1]
    # Adjust layout
    fig.suptitle(f'DL-Asm01-Task01-Linear Regression\nLearning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}\nFinal Train Loss: {final_train_loss:.4f}, Final Val Loss: {final_val_loss:.4f}', fontsize=16)
    plt.tight_layout()
    plt.show()

    


# Plot testing history
def plotTestHistory(test_history, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE):
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(np.arange(NUM_EPOCHS), test_history['loss'], label='Testing Loss', color='blue')
    
    plt.title('Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, NUM_EPOCHS+1, 10))
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    final_test_loss = test_history['loss'][-1]
    fig.suptitle(f'''DL-Asm01-Task01-Linear Regression\nLearning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, Num Epochs: {NUM_EPOCHS}\nFinal Test Loss: {final_test_loss:.4f}''', fontsize=16)
    plt.tight_layout()
    plt.show()



# Define the training of Linear Regression using Stochastic Gradient Descent with given lr, batch size
def train(X_train, y_train, X_val, y_val, model, lr, n_epochs, batch_size):
    m, n = X_train.shape
    theta = model
    # print(theta)
    train_losses = []
    train_history = {'loss': []}
    val_history = {'loss': []}

    for epoch in range(n_epochs):
        for i in range(0, m, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_hat_batch = feedForward(X_batch, theta)
            gradients = computeGradient(X_batch, y_batch, y_hat_batch)
            theta = optimSGD(lr, gradients, theta)
            train_losses.append(l2Loss(y_batch, y_hat_batch))
            # print(theta.shape)
        
        train_loss = np.mean(train_losses)
        train_history['loss'].append(train_loss)

        y_hat_val = feedForward(X_val, theta)
        val_loss = l2Loss(y_val, y_hat_val)
        val_history['loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

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
    test_history = {'loss': []}
    for epoch in range(n_epochs):
        test_loss = l2Loss(y_test, feedForward(X_test, theta))
        test_history['loss'].append(test_loss)
        print(f"Epoch {epoch+1}/{n_epochs} - Test Loss: {test_loss:.4f}")
    return test_history



def saveModel(theta_final, train_mean, train_variance, filename):
    with open(filename, 'wb') as f:
        pickle.dump((theta_final, train_mean, train_variance), f)


def loadModel(filename):
    with open(filename, 'rb') as f:
        theta_final, train_mean, train_variance = pickle.load(f)
    return theta_final, train_mean, train_variance



# Define root directories
ROOT_DIR = 'C:\\Users\\HP\\OneDrive\\Documents\\Univ Data\\ITU CS\\Deep Learning\\Assignments\\Assignment01\\Submitted\\Hassan_MSCS23001_01'



# Load the California housing dataset
dataset = fetch_california_housing()
X, y = dataset.data, dataset.target

print("Feature names are:")
print(dataset.feature_names[0:8])
print("Target name is: Median House Price")


# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=32)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, shuffle=False, random_state=32)

# Normalize the data
X_train_mean = np.mean(X_train, axis=0)
X_train_var = np.var(X_train, axis=0)
X_train_normalized = (X_train - X_train_mean) / X_train_var
X_val_normalized = (X_val - X_train_mean) / X_train_var
X_test_normalized = (X_test - X_train_mean) / X_train_var


# Initialize Linear Regression Model
model = linearRegressionModel(seed=132)


# Define the hyperparameters
LEARNING_RATE = 0.00001
NUM_EPOCHS = 100
BATCH_SIZE = 8


# Train the model using stochastic gradient descent with batch size
print("\n\n============================ TRAINING =========================================")
theta_final, train_history, val_history = train(X_train_normalized, y_train, X_val_normalized, y_val, model, LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE)


# Save the final model
model_save_name = ROOT_DIR + 'task01.pkl'
saveModel(theta_final, X_train_mean, X_train_var, model_save_name)
print(f'\nModel saved to drive: {model_save_name}')
print()


# Load the model for testing
loaded_model, train_mean, train_variance = loadModel(model_save_name)


# Test the model
print("\n\n============================ TESTING =========================================")
test_history = testModel(loaded_model, X_test_normalized, y_test, NUM_EPOCHS)


# Also calculate MSE and MAE of the test output
y_test_pred = X_test_normalized.dot(theta_final)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)


# Plot the training and validation loss history
plotHistory(train_history, val_history, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE)


# Plot test history
plotTestHistory(test_history, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE)

# Print evaluation metrics
print()
print("Testing MSE:", test_mse)
print("Testing MAE:", test_mae)
print()
