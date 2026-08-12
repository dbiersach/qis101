#!/usr/bin/env -S uv run
"""nn_primes_learn.py

Trains the network to decide whether an 8-bit integer is prime.

The target is encoded as 9 (binary 1001) for prime and 6 (binary 0110)
for composite, so the network has to drive four outputs rather than one.

TRAIN_COUNT controls how many of the 256 integers the network sees. The
rest are held back for nn_primes_test.py to score separately. See
nn_popcount_learn.py for the knobs worth experimenting with.
"""

from pathlib import Path

import numpy as np
from neural_network import SimpleNeuralNetwork, make_split

HIDDEN_SIZE = 16
TRAIN_COUNT = 240
EPOCHS = 10000
LEARNING_RATE = 0.01
SPLIT_SEED = 2024

OUTPUT_BITS = 4


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

    x, y = generate_data()
    for i in range(10):
        print(i, x[i], y[i])

    train_idx, test_idx = make_split(len(x), TRAIN_COUNT, SPLIT_SEED)
    print(f"\nTraining on {len(train_idx)} integers, holding back {len(test_idx)}")

    nn = SimpleNeuralNetwork(
        input_size=8, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_BITS
    )
    print(f"Network has {nn.parameter_count():,} weights\n")

    nn.train(x[train_idx], y[train_idx], epochs=EPOCHS, learning_rate=LEARNING_RATE)

    nn.save_model(
        Path(__file__).parent / "nn_primes_weights.npz",
        train_idx=train_idx,
        test_idx=test_idx,
    )


if __name__ == "__main__":
    main()
