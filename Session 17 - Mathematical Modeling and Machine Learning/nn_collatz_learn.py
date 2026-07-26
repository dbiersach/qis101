#!/usr/bin/env -S uv run
"""nn_collatz_learn.py

Trains the network to predict the Collatz stopping time of an integer,
using exactly the architecture that learns population count.

This one is expected to fail, and that is the point. Compare its
held-out score with nn_popcount_test.py and nn_primes_test.py.

Population count is a smooth function of the input bits: flip any single
bit and the answer moves by exactly 1. Collatz stopping time is not.
Flip one bit and the answer moves by 33 on average, and by as much as
121. Neighboring integers are unrelated:

    n = 26  ->  10 steps
    n = 27  -> 111 steps

There is no rule connecting the bit pattern to the stopping time for the
network to find, so nothing carries over to an integer it has not seen.
It can still memorize the training set perfectly, which is exactly what
makes the held-out score the honest measure.

Note the wider output: stopping times below 256 reach 127, so this lab
needs 7 output bits where population count and primality need 4.
"""

from pathlib import Path

import numpy as np
from neural_network import SimpleNeuralNetwork, make_split

HIDDEN_SIZE = 16
TRAIN_COUNT = 240
EPOCHS = 10000
LEARNING_RATE = 0.01
SPLIT_SEED = 2024

# Stopping times for n < 256 run 0..127, which needs 7 bits
OUTPUT_BITS = 7

# Start at 1: the Collatz sequence is not defined for 0
FIRST_N = 1
LAST_N = 255


def stop_time(n):
    """Return how many Collatz steps it takes for n to reach 1."""
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def generate_data():
    x = []
    y = []
    for i in range(FIRST_N, LAST_N + 1):
        # 8-bit binary representation of i (MSB->LSB)
        binary_row = [int(bit) for bit in format(i, "08b")]
        x.append(binary_row)
        # 7-bit binary representation of the stopping time
        binary_stop = [int(bit) for bit in format(stop_time(i), f"0{OUTPUT_BITS}b")]
        y.append(binary_stop)
    return np.array(x), np.array(y)


def main():
    np.random.seed(2024)

    x, y = generate_data()
    for i in range(10):
        print(FIRST_N + i, x[i], y[i], f"stop_time = {stop_time(FIRST_N + i)}")

    train_idx, test_idx = make_split(len(x), TRAIN_COUNT, SPLIT_SEED)
    print(f"\nTraining on {len(train_idx)} integers, holding back {len(test_idx)}")

    nn = SimpleNeuralNetwork(
        input_size=8, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_BITS
    )
    print(f"Network has {nn.parameter_count():,} weights\n")

    nn.train(x[train_idx], y[train_idx], epochs=EPOCHS, learning_rate=LEARNING_RATE)

    nn.save_model(
        Path(__file__).parent / "nn_collatz_weights.npz",
        train_idx=train_idx,
        test_idx=test_idx,
    )


if __name__ == "__main__":
    main()
