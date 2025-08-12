#!/usr/bin/env python3
"""nn_popcount_learn.py"""

import numpy as np
from neural_network import SimpleNeuralNetwork


def get_popcount(n):
    pop_count = 0
    while n > 0:
        pop_count += n % 2
        n //= 2
    return pop_count


def generate_data():
    x = []
    y = []
    for i in range(256):
        # 8-bit binary representation of i (MSB->LSB)
        binary_row = [int(bit) for bit in format(i, "08b")]
        x.append(binary_row)
        # 4-bit binary representation of the integer's population count
        binary_popcount = [int(bit) for bit in format(get_popcount(i), "04b")]
        y.append(binary_popcount)
    return np.array(x), np.array(y)


def main():
    np.random.seed(2024)

    # Generate training data
    x, y = generate_data()
    for i in range(10):
        print(i, x[i], y[i])

    # Create the neural network: 8 inputs, three hidden layers of 256 neurons, 4 outputs
    nn = SimpleNeuralNetwork(input_size=8, hidden_size=256, output_size=4)

    # Train the network with a specified learning rate
    nn.train(x, y, epochs=10000, learning_rate=0.01)

    # Save the weights as a Numpy NPZ file
    nn.save_model("nn_popcount_weights.npz")


if __name__ == "__main__":
    main()
