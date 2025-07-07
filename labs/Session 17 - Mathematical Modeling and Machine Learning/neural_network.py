#!/usr/bin/env python3
"""neural_network.py"""

import numpy as np


# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    # Note: x is assumed to be the output of relu
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # x is assumed to be the output of sigmoid
    return x * (1 - x)


# Neural network with 3 hidden layers
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = (
            hidden_size  # Each hidden layer will have hidden_size neurons
        )
        self.output_size = output_size

        # He initialization for layers with ReLU activation
        self.weights_input_hidden1 = np.random.randn(
            self.input_size, self.hidden_size
        ) * np.sqrt(2.0 / self.input_size)
        self.weights_hidden1_hidden2 = np.random.randn(
            self.hidden_size, self.hidden_size
        ) * np.sqrt(2.0 / self.hidden_size)
        self.weights_hidden2_hidden3 = np.random.randn(
            self.hidden_size, self.hidden_size
        ) * np.sqrt(2.0 / self.hidden_size)

        # Xavier initialization for output layer with sigmoid activation
        self.weights_hidden3_output = np.random.randn(
            self.hidden_size, self.output_size
        ) * np.sqrt(1.0 / self.hidden_size)

    def forward(self, x):
        # First hidden layer with ReLU activation
        self.hidden1_input = np.dot(x, self.weights_input_hidden1)
        self.hidden1_output = relu(self.hidden1_input)

        # Second hidden layer with ReLU activation
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2)
        self.hidden2_output = relu(self.hidden2_input)

        # Third hidden layer with ReLU activation
        self.hidden3_input = np.dot(self.hidden2_output, self.weights_hidden2_hidden3)
        self.hidden3_output = relu(self.hidden3_input)

        # Output layer with sigmoid activation
        self.final_input = np.dot(self.hidden3_output, self.weights_hidden3_output)
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, x, y, output, learning_rate):
        # Calculate output error and delta (using sigmoid derivative)
        self.loss = y - output
        self.output_delta = self.loss * sigmoid_derivative(output)

        # Backpropagate through third hidden layer (ReLU)
        self.hidden3_error = self.output_delta.dot(self.weights_hidden3_output.T)
        self.hidden3_delta = self.hidden3_error * relu_derivative(self.hidden3_output)

        # Backpropagate through second hidden layer (ReLU)
        self.hidden2_error = self.hidden3_delta.dot(self.weights_hidden2_hidden3.T)
        self.hidden2_delta = self.hidden2_error * relu_derivative(self.hidden2_output)

        # Backpropagate through first hidden layer (ReLU)
        self.hidden1_error = self.hidden2_delta.dot(self.weights_hidden1_hidden2.T)
        self.hidden1_delta = self.hidden1_error * relu_derivative(self.hidden1_output)

        # Update weights using the learning rate
        self.weights_hidden3_output += learning_rate * self.hidden3_output.T.dot(
            self.output_delta
        )
        self.weights_hidden2_hidden3 += learning_rate * self.hidden2_output.T.dot(
            self.hidden3_delta
        )
        self.weights_hidden1_hidden2 += learning_rate * self.hidden1_output.T.dot(
            self.hidden2_delta
        )
        self.weights_input_hidden1 += learning_rate * x.T.dot(self.hidden1_delta)

    def train(self, x, y, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:>4}, Error: {np.mean(np.abs(self.loss)):.5f}")


    def save_model(self, filename):
        np.savez_compressed(
            filename,
            weights_input_hidden1=self.weights_input_hidden1,
            weights_hidden1_hidden2=self.weights_hidden1_hidden2,
            weights_hidden2_hidden3=self.weights_hidden2_hidden3,
            weights_hidden3_output=self.weights_hidden3_output,
        )
        print(f"Model weights saved to {filename}")

    def load_model(self, filename):
        data = np.load(filename)
        self.weights_input_hidden1 = data["weights_input_hidden1"]
        self.weights_hidden1_hidden2 = data["weights_hidden1_hidden2"]
        self.weights_hidden2_hidden3 = data["weights_hidden2_hidden3"]
        self.weights_hidden3_output = data["weights_hidden3_output"]
        print(f"Model weights loaded from {filename}")


def main():
    print("This module is intended to be imported, not executed directly")


if __name__ == "__main__":
    main()
