# Neural Network in Java

This project implements a simple neural network in Java. It supports a feedforward network with a single hidden layer and uses backpropagation for training.

## Features
- Feedforward neural network
- Single hidden layer
- Sigmoid activation function
- Backpropagation for weight and bias updates
- Configurable learning rate
- Easily trainable with customizable number of epochs

## Project Structure

The `NeuralNetwork` class provides the implementation of the neural network, including methods for training, feedforward, and prediction.

### Main components:
- **Weights and Biases:** 
  - `weights1` and `weights2`: Weight matrices between input-hidden and hidden-output layers, respectively.
  - `bias1` and `bias2`: Bias vectors for hidden and output layers.
- **Activation Functions:** Sigmoid activation and its derivative are used.
- **Backpropagation:** For training, it calculates the gradients and updates weights and biases using the derivative of the error.

## Class Methods

### Constructor
```java
public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
```
- Initializes the network with the specified input, hidden, and output sizes.
- Randomly initializes the weights and biases using a Gaussian distribution.

### Training
```java
public void train(double[][] inputs, double[][] targets, int epochs)
```
- Trains the neural network using the backpropagation algorithm.
- Parameters:
  - `inputs`: 2D array of training data.
  - `targets`: 2D array of corresponding target labels.
  - `epochs`: Number of iterations for training.

### Prediction
```java
public double[] predict(double[] input)
```
- Feeds forward an input through the network and returns the predicted output.

### Feedforward
```java
private double[] feedForward(double[] input)
```
- Performs a forward pass through the network, calculating activations at each layer.

### Backpropagation
```java
private void backpropagation(double[] input, double[] target)
```
- Updates weights and biases based on the error calculated from the predicted output and target output.

## Getting Started

### Prerequisites
To compile and run this project, you will need:
- JDK 8 or higher

### Installation
Clone the repository:
```bash
git clone https://github.com/your-repo/java-neural-network.git
cd java-neural-network
```

### Downloading the MNIST Dataset

To train your network on the MNIST digit dataset, follow these steps to download and place the dataset in your project directory:

1. Download the MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/).

### Running the Code

You can create an instance of the `NeuralNetwork` class and train it as follows:
```java
public class Main {
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(784, 128, 10); // Example: 784 input nodes (28x28 pixels), 128 hidden, 10 output (digits 0-9)
        
        // Load and preprocess the MNIST dataset here
        
        double[][] inputs = {};  // Replace with actual MNIST input data
        double[][] targets = {}; // Replace with actual MNIST target data
        
        nn.train(inputs, targets, 10000); // Train the network for 10,000 epochs
        double[] prediction = nn.predict(new double[] {}); // Replace with an actual input example
        System.out.println("Prediction: " + Arrays.toString(prediction));
    }
}
```

### Example Output

During training, the console will print each epoch's completion:
```
Epoch 0 completed
Epoch 1 completed
...
Epoch 9999 completed
```

Predictions will be printed at the end of the training process:
```
Prediction: [0.923652] // Example
```

## Customization

- **Input/Hidden/Output Sizes:** Modify the number of nodes in the constructor.
- **Learning Rate:** Adjust the `learningRate` parameter to control how fast the network learns.
- **Epochs:** Customize the number of training epochs for better accuracy.
