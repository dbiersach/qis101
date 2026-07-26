#!/usr/bin/env -S uv run
"""neural_network.py

A small feed-forward network with three hidden layers, shared by the
nn_popcount, nn_primes, and nn_collatz labs.

Also provides make_split(), so a lab can train on part of its data and
hold the rest back. A network that scores well on data it trained on has
only memorized; the held-out score is the one that shows whether it
learned anything general.
"""

import numpy as np

# Keys that describe the architecture or the weights themselves. Anything
# else found in a saved file is extra data the caller stored alongside it.
ARCH_KEYS = ("input_size", "hidden_size", "output_size")
WEIGHT_KEYS = (
    "weights_input_hidden1",
    "weights_hidden1_hidden2",
    "weights_hidden2_hidden3",
    "weights_hidden3_output",
)


# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    # Note: x is assumed to be the output of relu
    return (x > 0).astype(float)


def sigmoid(x):
    """Logistic sigmoid, evaluated so that exp() cannot overflow.

    The direct form 1 / (1 + exp(-x)) overflows once x is about -710,
    because exp(-x) then exceeds the largest float64. For negative x the
    algebraically identical form exp(x) / (1 + exp(x)) is used instead,
    where the exponent stays negative and exp(x) never exceeds 1.
    """
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)

    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))

    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1 + exp_x)

    return result


def sigmoid_derivative(x):
    # x is assumed to be the output of sigmoid
    return x * (1 - x)


def make_split(sample_count, train_count, seed):
    """Split sample_count items into a training set and a held-out set.

    Parameters
    ----------
    sample_count : int
        Total number of samples available.
    train_count : int
        How many to train on. The remainder are held out. Pass
        sample_count to train on everything and hold nothing back.
    seed : int
        Fixes which samples land in which set, so the learn and test
        scripts of a lab always agree on the split.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Indices of the training samples and of the held-out samples.
    """
    if not 0 < train_count <= sample_count:
        raise ValueError(f"train_count must be in 1..{sample_count}, got {train_count}")
    order = np.random.default_rng(seed).permutation(sample_count)
    return order[:train_count], order[train_count:]


# Neural network with 3 hidden layers
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size  # Each hidden layer has hidden_size neurons
        self.output_size = output_size

        # He initialization for layers with ReLU activation
        self.weights_input_hidden1 = np.random.randn(
            self.input_size, self.hidden_size
        ) * np.sqrt(2.0 / self.input_size)
        self.weights_hidden1_hidden2 = np.random.randn(
            self.hidden_size, self.hidden_size
        ) * np.sqrt(2.0 / self.hidden_size)
        self.weights_hidden2_hidden3 = np.random.randn(
            self.hidden_size, self.hidden_size
        ) * np.sqrt(2.0 / self.hidden_size)

        # Xavier initialization for output layer with sigmoid activation
        self.weights_hidden3_output = np.random.randn(
            self.hidden_size, self.output_size
        ) * np.sqrt(1.0 / self.hidden_size)

    def parameter_count(self):
        """Return how many weights the network holds."""
        return (
            self.weights_input_hidden1.size
            + self.weights_hidden1_hidden2.size
            + self.weights_hidden2_hidden3.size
            + self.weights_hidden3_output.size
        )

    def forward(self, x):
        # First hidden layer with ReLU activation
        self.hidden1_input = np.dot(x, self.weights_input_hidden1)
        self.hidden1_output = relu(self.hidden1_input)

        # Second hidden layer with ReLU activation
        self.hidden2_input = np.dot(self.hidden1_output, self.weights_hidden1_hidden2)
        self.hidden2_output = relu(self.hidden2_input)

        # Third hidden layer with ReLU activation
        self.hidden3_input = np.dot(self.hidden2_output, self.weights_hidden2_hidden3)
        self.hidden3_output = relu(self.hidden3_input)

        # Output layer with sigmoid activation
        self.final_input = np.dot(self.hidden3_output, self.weights_hidden3_output)
        self.final_output = sigmoid(self.final_input)

        return self.final_output

    def backward(self, x, y, output, learning_rate):
        # Calculate output error and delta (using sigmoid derivative)
        self.loss = y - output
        self.output_delta = self.loss * sigmoid_derivative(output)

        # Backpropagate through third hidden layer (ReLU)
        self.hidden3_error = self.output_delta.dot(self.weights_hidden3_output.T)
        self.hidden3_delta = self.hidden3_error * relu_derivative(self.hidden3_output)

        # Backpropagate through second hidden layer (ReLU)
        self.hidden2_error = self.hidden3_delta.dot(self.weights_hidden2_hidden3.T)
        self.hidden2_delta = self.hidden2_error * relu_derivative(self.hidden2_output)

        # Backpropagate through first hidden layer (ReLU)
        self.hidden1_error = self.hidden2_delta.dot(self.weights_hidden1_hidden2.T)
        self.hidden1_delta = self.hidden1_error * relu_derivative(self.hidden1_output)

        # Update weights using the learning rate
        self.weights_hidden3_output += learning_rate * self.hidden3_output.T.dot(
            self.output_delta
        )
        self.weights_hidden2_hidden3 += learning_rate * self.hidden2_output.T.dot(
            self.hidden3_delta
        )
        self.weights_hidden1_hidden2 += learning_rate * self.hidden1_output.T.dot(
            self.hidden2_delta
        )
        self.weights_input_hidden1 += learning_rate * x.T.dot(self.hidden1_delta)

    def train(self, x, y, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, output, learning_rate)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:>5}, Error: {np.mean(np.abs(self.loss)):.5f}")

    def save_model(self, filename, **extra):
        """Save the weights, the layer sizes, and any extra arrays given.

        The layer sizes are stored so that load_network() can rebuild a
        network of the right shape. The labs use **extra to record which
        samples were trained on and which were held out.
        """
        np.savez_compressed(
            filename,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            weights_input_hidden1=self.weights_input_hidden1,
            weights_hidden1_hidden2=self.weights_hidden1_hidden2,
            weights_hidden2_hidden3=self.weights_hidden2_hidden3,
            weights_hidden3_output=self.weights_hidden3_output,
            **extra,
        )
        print(f"Model weights saved to {filename}")

    def load_model(self, filename):
        """Load the weights and return whatever extra arrays were saved."""
        data = np.load(filename)
        self.weights_input_hidden1 = data["weights_input_hidden1"]
        self.weights_hidden1_hidden2 = data["weights_hidden1_hidden2"]
        self.weights_hidden2_hidden3 = data["weights_hidden2_hidden3"]
        self.weights_hidden3_output = data["weights_hidden3_output"]
        print(f"Model weights loaded from {filename}")
        return {k: data[k] for k in data.files if k not in ARCH_KEYS + WEIGHT_KEYS}


def load_network(filename):
    """Rebuild a saved network, sized to match the file.

    Returns the network and a dict of whatever extra arrays were stored
    with it. Using this rather than constructing the network by hand
    means a test script cannot disagree with its learn script about the
    hidden layer size.
    """
    data = np.load(filename)
    network = SimpleNeuralNetwork(
        input_size=int(data["input_size"]),
        hidden_size=int(data["hidden_size"]),
        output_size=int(data["output_size"]),
    )
    return network, network.load_model(filename)


def main():
    print("This module is intended to be imported, not executed directly")

    # Quick check that the stable sigmoid handles what the plain form cannot
    extreme = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    print(f"sigmoid({extreme}) =\n  {sigmoid(extreme)}")
    print("The plain 1 / (1 + exp(-x)) form overflows at x = -800")


if __name__ == "__main__":
    main()
