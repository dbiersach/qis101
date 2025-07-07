#!/usr/bin/env python3
"""nn_xor.py"""

import numpy as np


# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Neural network class
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly with mean 0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, x):
        # Forward propagation through our network
        self.hidden_input = np.dot(x, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, x, y, output):
        # Calculate the error
        self.loss = y - output
        self.output_delta = self.loss * sigmoid_derivative(output)

        # How much did each hidden layer neuron contribute to the output error?
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * sigmoid_derivative(self.hidden_output)

        # Update the weights
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta)
        self.weights_input_hidden += x.T.dot(self.hidden_delta)

    def train(self, x, y, epochs=10_000):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output)
            # Optional: adjust the learning rate or print the error every n epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:>4}, Error: {np.mean(np.abs(self.loss)):.5f}")


def main():
    # Input dataset
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Output dataset
    y = np.array([[0], [1], [1], [0]])

    # Create the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    nn.train(x, y)

    # Test the neural network
    nn_final = nn.forward(x)
    for i in range(4):
        print(x[i], y[i], nn_final[i])


if __name__ == "__main__":
    main()
