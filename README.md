
# FashionNet: Fashion-MNIST Image Classification with PyTorch

FashionNet is a deep learning project that classifies images from the Fashion-MNIST dataset using a Convolutional Neural Network (CNN) implemented in PyTorch. This project aims to develop a robust image classifier capable of distinguishing between various clothing items, using CNN layers to extract features effectively.

# Project Overview

- Objective: Build a CNN to classify Fashion-MNIST images into 10 categories, enhancing skills in computer vision and deep learning using PyTorch.
- Dataset: The Fashion-MNIST dataset, containing 60,000 training images and 10,000 test images across 10 clothing categories, including t-shirts, trousers, and shoes.

# Key Features

1. Model Architecture (FashionNet):
   - Two convolutional layers with ReLU activations, followed by max pooling and dropout layers.
   - Dropout rates of 0.25 and 0.5 to reduce overfitting.
   - Two fully connected layers for feature vector transformation and class prediction.
   - Log softmax output layer to compute class probabilities.

2. Training Process:
   - Cross-entropy loss function (F.nll_loss) and Adam optimizer for weight updates.
   - Includes a learning rate scheduler (StepLR) to optimize the learning process.
   - Trained over multiple epochs with training loss and accuracy printed at each step.

3. Evaluation and Visualization:
   - Model evaluation on the test dataset to calculate overall accuracy.
   - `show()` function displays a grid of test images with predicted vs. true labels.


# Usage

1. Run the Notebook:
   Open `Fashion_MNIST_Image_Classification.ipynb` in Jupyter Notebook.
   
2. Train and Test the Model:
   Follow the steps in the notebook to define, train, evaluate, and visualize model predictions on Fashion-MNIST.

# Results and Discussion

- Accuracy: The trained model provides accurate predictions across the 10 fashion categories.
- Differences with MNIST: The Fashion-MNIST dataset includes complex textures and shapes compared to MNIST, making it more challenging.

# Future Enhancements

- Experiment with Deeper Architectures: Test with additional layers or different architectures to improve accuracy.
- Advanced Regularization: Explore techniques like batch normalization to enhance model robustness.
