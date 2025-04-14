#!/usr/bin/env python3
"""nn_primes_learn.py"""

import numpy as np
from neural_network import SimpleNeuralNetwork


def is_prime(n):
    if n == 0 or n == 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    else:
        return all(n % factor != 0 for factor in range(3, int(np.sqrt(n)) + 1, 2))


def generate_data():
    x = []
    y = []
    for i in range(256):
        # 8-bit binary representation of i (MSB->LSB)
        binary_row = [int(bit) for bit in format(i, "08b")]
        x.append(binary_row)
        # 4-bit binary representation of primality
        if is_prime(i):
            y.append([1, 0, 0, 1])  # = 9 if i is prime
        else:
            y.append([0, 1, 1, 0])  # = 6 if i is composite
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
    nn.save_model("nn_primes_weights.npz")


if __name__ == "__main__":
    main()
