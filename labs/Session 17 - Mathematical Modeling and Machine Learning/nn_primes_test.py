#!/usr/bin/env python3
"""nn_primes_test.py"""

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
    for i in range(256):
        # 8-bit binary representation of i
        binary_row = [int(bit) for bit in format(i, "08b")]
        x.append(binary_row)
    return np.array(x)


def print_model(nn):
    print(f"\n{nn.weights_input_hidden1.shape = }")
    print(f"{nn.weights_input_hidden1[:4, :4]}\n")

    print(f"{nn.weights_hidden1_hidden2.shape = }")
    print(f"{nn.weights_hidden1_hidden2[:4, :4]}\n")

    print(f"{nn.weights_hidden2_hidden3.shape = }")
    print(f"{nn.weights_hidden2_hidden3[:4, :4]}\n")

    print(f"{nn.weights_hidden3_output.shape = }")
    print(f"{nn.weights_hidden3_output[:4, :4]}\n")


def main():
    # Generate training data
    x = generate_data()

    nn = SimpleNeuralNetwork(input_size=8, hidden_size=256, output_size=4)
    nn.load_model("nn_primes_weights.npz")
    print_model(nn)

    # Evaluate the quality of the trained network
    final = nn.forward(x)
    no_errors = True
    for i in range(256):
        v = 0
        for j in range(4):
            v += int(round(final[i, j], 0) * 2 ** (3 - j))
        predicted_is_prime = v == 9
        actual_is_prime = is_prime(i)
        if predicted_is_prime != actual_is_prime:
            print(f"Error with {i}:", end=" ")
            print(f"Predicted 'Is Prime'= {predicted_is_prime}", end=", ")
            print(f"Actual 'Is Prime'= {actual_is_prime}")
            no_errors = False
    if no_errors:
        print("The model identified all primes")


if __name__ == "__main__":
    main()
