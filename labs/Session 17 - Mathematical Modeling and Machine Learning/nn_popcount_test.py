#!/usr/bin/env python3
"""nn_popcount_test.py"""

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
    for i in range(256):
        # 8-bit binary representation of i
        binary_row = [int(bit) for bit in format(i, "08b")]
        x.append(binary_row)
    return np.array(x)


def main():
    # Generate training data
    x = generate_data()

    nn = SimpleNeuralNetwork(input_size=8, hidden_size=256, output_size=4)
    nn.load_model("nn_popcount_weights.npz")

    # Evaluate the quality of the trained network
    final = nn.forward(x)
    no_errors = True
    for i in range(256):
        predicted_pop_count = 0
        for j in range(4):
            predicted_pop_count += int(round(final[i, j], 0) * 2 ** (3 - j))
        actual_pop_count = get_popcount((i))
        if predicted_pop_count != actual_pop_count:
            print(f"Error with {i}:", end=" ")
            print(f"Predicted Popcount: {predicted_pop_count}", end=", ")
            print(f"Actual Popcount: {actual_pop_count}")
            no_errors = False
    if no_errors:
        print("The model identified all primes")


if __name__ == "__main__":
    main()
