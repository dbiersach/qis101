#!/usr/bin/env -S uv run
"""nn_popcount_learn.py

Trains the network to count the 1 bits in an 8-bit integer.

TRAIN_COUNT controls how many of the 256 integers the network sees. The
rest are held back, so nn_popcount_test.py can report separately on the
integers that were trained on and the ones that were not.

Things worth changing and re-running:
    TRAIN_COUNT = 256   trains on everything, holds nothing back, and so
                        can only ever demonstrate memorization
    HIDDEN_SIZE = 256   134,144 weights for 240 examples, which is more
                        than enough to memorize the answers instead of
                        learning the rule
"""

from pathlib import Path

import numpy as np
from neural_network import SimpleNeuralNetwork, make_split

HIDDEN_SIZE = 16
TRAIN_COUNT = 240
EPOCHS = 10000
LEARNING_RATE = 0.01
SPLIT_SEED = 2024

# Population count of an 8-bit integer runs 0..8, which needs 4 bits
OUTPUT_BITS = 4


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

    x, y = generate_data()
    for i in range(10):
        print(i, x[i], y[i])

    train_idx, test_idx = make_split(len(x), TRAIN_COUNT, SPLIT_SEED)
    print(f"\nTraining on {len(train_idx)} integers, holding back {len(test_idx)}")

    nn = SimpleNeuralNetwork(
        input_size=8, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_BITS
    )
    print(f"Network has {nn.parameter_count():,} weights\n")

    # Train on the training split only - the held-out integers are never seen
    nn.train(x[train_idx], y[train_idx], epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # Save the weights next to this script, along with the split, so the
    # matching test script scores exactly the integers that were held back
    nn.save_model(
        Path(__file__).parent / "nn_popcount_weights.npz",
        train_idx=train_idx,
        test_idx=test_idx,
    )


if __name__ == "__main__":
    main()
