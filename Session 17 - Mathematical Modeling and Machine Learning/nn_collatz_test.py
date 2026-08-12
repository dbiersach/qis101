#!/usr/bin/env -S uv run
"""nn_collatz_test.py

Scores the trained network on the integers it saw and on the integers it
did not, and lists every held-out prediction next to the true answer.

Because a stopping time is a magnitude rather than a label, this lab also
reports the average size of the error, and compares it with two baselines
that do no learning at all:

    always answer the median of the training stopping times
    copy the answer of the closest training integer in bit space

If the network cannot beat those, it has found no usable rule.
"""

from pathlib import Path

import numpy as np
from neural_network import load_network

OUTPUT_BITS = 7
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
    for i in range(FIRST_N, LAST_N + 1):
        # 8-bit binary representation of i
        binary_row = [int(bit) for bit in format(i, "08b")]
        x.append(binary_row)
    return np.array(x)


def decode(rows):
    """Read each row of sigmoid outputs back as an integer."""
    place_value = 2 ** np.arange(OUTPUT_BITS - 1, -1, -1)
    return (np.round(rows) * place_value).sum(axis=1).astype(int)


def score(name, nn, x, indices, actual):
    """Print how many stopping times are exactly right, and by how much
    the rest are off."""
    if len(indices) == 0:
        print(f"{name:<12} (none held back)")
        return
    predicted = decode(nn.forward(x[indices]))
    correct = predicted == actual[indices]
    error = np.abs(predicted - actual[indices]).mean()
    print(
        f"{name:<12} {correct.sum():>3}/{len(indices):<3} exact"
        f"  ({100 * correct.mean():>5.1f}%)   average error {error:>5.1f} steps"
    )


def main():
    x = generate_data()
    integers = np.arange(FIRST_N, LAST_N + 1)
    actual = np.array([stop_time(int(n)) for n in integers])

    # The weights are not stored in the repository, so they have to be
    # trained locally before this script has anything to load
    weights_path = Path(__file__).parent / "nn_collatz_weights.npz"
    if not weights_path.exists():
        print(f"Cannot find {weights_path.name}")
        print("Run nn_collatz_learn.py first to train the network")
        print("and save its weights, then run this script again.")
        return

    nn, saved = load_network(weights_path)
    train_idx, test_idx = saved["train_idx"], saved["test_idx"]
    print(
        f"Hidden layers of {nn.hidden_size} neurons, {nn.parameter_count():,} weights"
    )

    print("\nCollatz stopping time")
    score("trained on", nn, x, train_idx, actual)
    score("held out", nn, x, test_idx, actual)

    if len(test_idx) == 0:
        print("\nNothing was held back, so this run can only show memorization.")
        return

    # Two baselines that involve no learning whatsoever
    median = int(np.median(actual[train_idx]))
    median_error = np.abs(median - actual[test_idx]).mean()
    print(
        f"{'baseline':<12} always answer {median:>3}"
        f"            average error {median_error:>5.1f} steps"
    )

    nearest_error = []
    for i in test_idx:
        # Hamming distance: how many bits differ between the two integers
        distance = [
            (int(integers[i]) ^ int(integers[j])).bit_count() for j in train_idx
        ]
        nearest = actual[train_idx[int(np.argmin(distance))]]
        nearest_error.append(abs(nearest - actual[i]))
    print(
        f"{'baseline':<12} copy nearest neighbor   average error"
        f" {np.mean(nearest_error):>5.1f} steps"
    )

    print("\nEvery held-out integer:")
    print(f"  {'n':>4} {'actual':>7} {'predicted':>10} {'error':>7}")
    predicted = decode(nn.forward(x[test_idx]))
    for n, a, p in sorted(zip(integers[test_idx], actual[test_idx], predicted)):
        print(f"  {n:>4} {a:>7} {p:>10} {p - a:>+7}")


if __name__ == "__main__":
    main()
