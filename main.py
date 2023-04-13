import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from string import ascii_lowercase
import matplotlib.pyplot as plt

def generate_dataset(font_paths, size=28):
    alphabet = ascii_lowercase
    images = []
    labels = []

    for label, letter in enumerate(alphabet):
        for font_path in font_paths:
            font = ImageFont.truetype(str(font_path), size)
            image = Image.new('L', (size, size), color=255)
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), letter, font=font, fill=0)
            data = np.array(image).flatten() / 255.0
            images.append(data)
            labels.append(label)

    return np.array(images), np.array(labels, dtype=int)  # Convert labels to integers explicitly

# Get installed fonts
def get_installed_fonts():
    font_dir = "/usr/share/fonts/truetype"  # Change this to the font directory on your system
    if os.name == "nt":
        font_dir = "C:\\Windows\\Fonts"
    
    font_paths = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf"):
                font_paths.append(os.path.join(root, file))
    return font_paths

# Load fonts and generate dataset
font_paths = get_installed_fonts()
X, y = generate_dataset(font_paths)

# Preprocess the data
def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def split_data(X, y, ratio=0.8):
    num_train = int(ratio * len(X))
    indices = np.random.permutation(len(X))
    train_indices, test_indices = indices[:num_train], indices[num_train:]
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X, y)
y_train_encoded = one_hot_encode(y_train, 26)
y_test_encoded = one_hot_encode(y_test, 26)

class ANN:
    def __init__(self, input_size, hidden_size1, hidden_size2=None, output_size=26):
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        if hidden_size2:
            self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
            self.b2 = np.zeros((1, hidden_size2))
            self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
            self.b3 = np.zeros((1, output_size))
        else:
            self.W2 = np.random.randn(hidden_size1, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))
            self.W3 = None
            self.b3 = None

    def forward(self, X):
        # First hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU

        if self.W3 is not None:
            # Second hidden layer
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = np.maximum(0, self.z2)  # ReLU

            # Output layer
            self.z3 = np.dot(self.a2, self.W3) + self.b3
            self.output = np.exp(self.z3) / np.sum(np.exp(self.z3), axis=1, keepdims=True)
        else:
            # Output layer
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.output = np.exp(self.z2) / np.sum(np.exp(self.z2), axis=1, keepdims=True)

    def compute_loss(self, y_true):
        num_samples = y_true.shape[0]
        correct_logprobs = -np.log(self.output[range(num_samples), y_true])
        loss = np.sum(correct_logprobs) / num_samples
        return loss

    def backward(self, X, y_true):
        num_samples = X.shape[0]
        y_true_encoded = one_hot_encode(y_true, self.output.shape[1])

        if self.W3 is not None:
            # Backpropagation for output layer
            delta3 = self.output
            delta3[range(num_samples), y_true] -= 1
            delta3 /= num_samples

            # Backpropagation for second hidden layer
            dW3 = np.dot(self.a2.T, delta3)
            db3 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = np.dot(delta3, self.W3.T) * (self.a2 > 0)

            # Backpropagation for first hidden layer
            dW2 = np.dot(self.a1.T, delta2)
            db2 = np.sum(delta2, axis=0, keepdims=True)
            delta1 = np.dot(delta2, self.W2.T) * (self.a1 > 0)

        else:
            # Backpropagation for output layer
            delta2 = self.output
            delta2[range(num_samples), y_true] -= 1
            delta2 /= num_samples

            # Backpropagation for first hidden layer
            dW2 = np.dot(self.a1.T, delta2)
            db2 = np.sum(delta2, axis=0, keepdims=True)
            delta1 = np.dot(delta2, self.W2.T) * (self.a1 > 0)

        # Backpropagation for input layer
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Update parameters
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        if self.W3 is not None:
            self.W3 -= self.learning_rate * dW3
            self.b3 -= self.learning_rate * db3

    def train(self, X, y, learning_rate=0.005, epochs=1000, print_loss=True):
        self.learning_rate = learning_rate
        self.losses = []

        for epoch in range(epochs):
            self.forward(X)
            loss = self.compute_loss(y)
            self.losses.append(loss)
            self.backward(X, y)

            if print_loss and epoch % 100 == 0:
                print(f'Loss at epoch {epoch}: {loss}')

    def predict(self, X):
        self.forward(X)
        predictions = np.argmax(self.output, axis=1)
        return predictions

# Train ANN models with 1 and 2 hidden layers
ann1 = ANN(input_size=28*28, hidden_size1=64, output_size=26)
ann1.train(X_train, y_train, learning_rate=0.1, epochs=2000) # earning_rate=0.1, epochs=2000 get accuracy 69%

ann2 = ANN(input_size=28*28, hidden_size1=64, hidden_size2=32, output_size=26)
ann2.train(X_train, y_train, learning_rate=0.15, epochs=3500) # earning_rate=0.15, epochs=3500 get accuracy 63%

def compute_accuracy(predictions, y_true):
    correct = np.sum(predictions == y_true)
    total = len(y_true)
    return correct / total

# Test ANN models with 1 and 2 hidden layers
predictions1 = ann1.predict(X_test)
accuracy1 = compute_accuracy(predictions1, y_test)
print(f'Accuracy of ANN model with 1 hidden layer: {accuracy1 * 100:.2f}%')

predictions2 = ann2.predict(X_test)
accuracy2 = compute_accuracy(predictions2, y_test)
print(f'Accuracy of ANN model with 2 hidden layers: {accuracy2 * 100:.2f}%')

def visualize_predictions(X, y_true, predictions, num_samples=10):
    indices = np.random.choice(len(X), num_samples)
    X_subset = X[indices]
    y_true_subset = y_true[indices]
    predictions_subset = predictions[indices]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    for i, (img, true_label, predicted_label) in enumerate(zip(X_subset, y_true_subset, predictions_subset)):
        ax = axes[i]
        img = img.reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'True: {ascii_lowercase[true_label]}\nPredicted: {ascii_lowercase[predicted_label]}')
    plt.show()

# Visualize predictions for ANN models with 1 and 2 hidden layers
print("Predictions of ANN model with 1 hidden layer:")
visualize_predictions(X_test, y_test, predictions1)

print("Predictions of ANN model with 2 hidden layers:")
visualize_predictions(X_test, y_test, predictions2)