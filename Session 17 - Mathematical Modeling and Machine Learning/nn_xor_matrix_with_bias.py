#!/usr/bin/env -S uv run
"""nn_xor_matrix_with_bias.py"""

from enum import Enum, auto
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm


class InitStrategy(Enum):
    RND_NORMAL = auto()
    XAVIER = auto()


# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def init_weights(topology, init_strategy):
    input_size, hidden_size, output_size = topology

    match init_strategy:
        case InitStrategy.RND_NORMAL:
            np.random.seed(2020)
            W_ih = np.random.randn(input_size, hidden_size)
            W_ho = np.random.randn(hidden_size, output_size)

        case InitStrategy.XAVIER:
            # Xavier Glorot & Yoshua Bengio (2010) - Uniform Distribution
            limit = np.sqrt(6 / (input_size + hidden_size))
            W_ih = np.random.uniform(-limit, limit, size=(input_size, hidden_size))
            limit = np.sqrt(6 / (hidden_size + output_size))
            W_ho = np.random.uniform(-limit, limit, size=(hidden_size, output_size))

    # Biases initialized to zeros
    b_ih = np.zeros((1, hidden_size))
    b_ho = np.zeros((1, output_size))

    return W_ih, W_ho, b_ih, b_ho


def train(init_strategy, learning_rate=1.0):
    # Input and output for XOR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Neural Network Topology
    input_size, hidden_size, output_size = 2, 4, 1
    topology = input_size, hidden_size, output_size

    # Initialize weights and biases
    W_ih, W_ho, b_ih, b_ho = init_weights(topology, init_strategy)

    # Training loop
    for _ in range(EPOCHS):
        # Forward pass
        hidden_input = np.dot(x, W_ih) + b_ih
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, W_ho) + b_ho
        final_output = sigmoid(final_input)

        # Compute loss function
        residual = y - final_output
        loss = 0.5 * residual**2

        # Backpropagate loss and update weights
        output_delta = residual * sigmoid_derivative(final_output)
        W_ho += learning_rate * np.dot(hidden_output.T, output_delta)

        hidden_error = np.dot(output_delta, W_ho.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
        W_ih += learning_rate * np.dot(x.T, hidden_delta)

        # Update biases
        b_ih += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
        b_ho += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

    final_hidden_output = sigmoid(np.dot(x, W_ih) + b_ih)
    final_output = sigmoid(np.dot(final_hidden_output, W_ho) + b_ho)
    final_loss = np.mean(np.abs(loss))

    return final_loss, final_output


def run_model(init_strategy):
    final_loss, final_output = train(init_strategy)
    print(f"\nInitial Weights Strategy: {init_strategy.name}")
    print(f"Final Output: {final_output.T}")
    print(f"Final Loss: {final_loss:.5f}\n")


def plot_models():
    learning_rate = np.linspace(0.1, 1.0, 100)
    final_loss_rnd = np.zeros_like(learning_rate)
    final_loss_xavier = np.zeros_like(learning_rate)

    for i in tqdm(range(len(final_loss_rnd)), desc="Training"):
        final_loss_rnd[i] = train(InitStrategy.RND_NORMAL, learning_rate[i])[0]
        final_loss_xavier[i] = train(InitStrategy.XAVIER, learning_rate[i])[0]

    plt.figure(Path(__file__).name)
    plt.plot(learning_rate, final_loss_rnd, label="RND Normal")
    plt.plot(learning_rate, final_loss_xavier, label="Xavier")
    plt.title(f"XOR NN (2-4-1) with Bias ({EPOCHS:,} Epochs)\nLearning Rate vs. Loss")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Loss")
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.legend()
    plt.grid()
    plt.show()


def main():
    global EPOCHS
    EPOCHS = 10_000

    run_model(InitStrategy.RND_NORMAL)
    run_model(InitStrategy.XAVIER)

    plot_models()


if __name__ == "__main__":
    main()
