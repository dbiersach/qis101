#!/usr/bin/env -S uv run
"""nn_popcount_test.py

Scores the trained network twice: once on the integers it was trained on
and once on the integers held back. The first number says whether it
memorized, the second whether it learned.
"""

from pathlib import Path

import numpy as np
from neural_network import load_network

OUTPUT_BITS = 4


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


def decode(rows):
    """Read each row of sigmoid outputs back as an integer."""
    place_value = 2 ** np.arange(OUTPUT_BITS - 1, -1, -1)
    return (np.round(rows) * place_value).sum(axis=1).astype(int)


def score(name, nn, x, indices, actual):
    """Print how many of the given integers the network gets right."""
    if len(indices) == 0:
        print(f"{name:<12} (none held back)")
        return
    predicted = decode(nn.forward(x[indices]))
    correct = predicted == actual[indices]
    print(
        f"{name:<12} {correct.sum():>3}/{len(indices):<3} correct"
        f"  ({100 * correct.mean():>5.1f}%)"
    )
    for i in np.array(indices)[~correct][:10]:
        print(
            f"    n={i:>3}  predicted {decode(nn.forward(x[[i]]))[0]:>2},"
            f"  actual {actual[i]:>2}"
        )


def main():
    x = generate_data()
    actual = np.array([get_popcount(i) for i in range(256)])

    # The weights are not stored in the repository, so they have to be
    # trained locally before this script has anything to load
    weights_path = Path(__file__).parent / "nn_popcount_weights.npz"
    if not weights_path.exists():
        print(f"Cannot find {weights_path.name}")
        print("Run nn_popcount_learn.py first to train the network")
        print("and save its weights, then run this script again.")
        return

    nn, saved = load_network(weights_path)
    print(
        f"Hidden layers of {nn.hidden_size} neurons, {nn.parameter_count():,} weights"
    )

    print("\nPopulation count")
    score("trained on", nn, x, saved["train_idx"], actual)
    score("held out", nn, x, saved["test_idx"], actual)


if __name__ == "__main__":
    main()
