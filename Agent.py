import random
from collections.abc import Callable

from Brain import Brain, Neuron, sigmoid

class Agent:
    @staticmethod
    def _create_brain(input_size: int, hidden_layer_sizes: list[int], output_size: int, activation_function: Callable[[float], float]) -> Brain:
        prev_size = input_size
        hidden_layers = []
        for layer_size in hidden_layer_sizes:
            layer = []
            for _ in range(layer_size):
                weights = [random.gauss(0, 1) for _ in range(prev_size)]
                bias = random.gauss(0, 1)
                layer.append(Neuron(weights, bias))
            hidden_layers.append(layer)
            prev_size = layer_size

        output_layer = []
        for _ in range(output_size):
            weights = [random.gauss(0, 1) for _ in range(prev_size)]
            bias = random.gauss(0, 1)
            output_layer.append(Neuron(weights, bias))
        return Brain(input_size, hidden_layers, output_layer, activation_function)


    def __init__(self, brain: Brain):
        self.brain = brain

    @classmethod
    def new(cls, input_size: int, hidden_layers: list[int], output_size: int, activation_function=None):
        if activation_function is None:
            activation_function = sigmoid

        return cls(cls._create_brain(input_size, hidden_layers, output_size, activation_function))

    def copy(self):
        return Agent(self.brain.copy())

    def learn_random(self):
        for layer in self.brain.hiddenLayers:
            for neuron in layer:
                neuron.weights = [w + random.gauss(0, 0.1) for w in neuron.weights]
                neuron.bias += random.gauss(0, 0.1)
        for neuron in self.brain.outputLayer:
            neuron.weights = [w + random.gauss(0, 0.1) for w in neuron.weights]
            neuron.bias += random.gauss(0, 0.1)

    def backpropagate(self, input_data: list[float], target_output: list[float], learning_rate: float):
        self.brain.set_input(input_data)
        self.brain._forward_propagate()

        output_deltas = []
        for i, neuron in enumerate(self.brain.outputLayer):
            error = target_output[i] - neuron.output
            delta = error * neuron.output * (1 - neuron.output)
            output_deltas.append(delta)
            for j in range(len(neuron.weights)):
                neuron.weights[j] += learning_rate * delta * self.brain.hiddenLayers[-1][j].output
            neuron.bias += learning_rate * delta

        current_deltas = output_deltas
        current_weights = [n.weights for n in self.brain.outputLayer]

        for i in reversed(range(len(self.brain.hiddenLayers))):
            layer = self.brain.hiddenLayers[i]
            next_deltas = []

            for j, neuron in enumerate(layer):
                error = sum(current_weights[k][j] * current_deltas[k] for k in range(len(current_deltas)))
                delta = error * neuron.output * (1 - neuron.output)
                next_deltas.append(delta)

                for k in range(len(neuron.weights)):
                    if i == 0:
                        neuron.weights[k] += learning_rate * delta * self.brain.inputLayer[k]
                    else:
                        neuron.weights[k] += learning_rate * delta * self.brain.hiddenLayers[i - 1][k].output
                neuron.bias += learning_rate * delta

            current_deltas = next_deltas
            current_weights = [n.weights for n in layer]