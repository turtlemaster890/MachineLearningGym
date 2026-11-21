import random
from collections.abc import Callable

from Brain import ActivationFunction, Brain, ConvolutionalBrain, Image, Layer, Neuron, SIGMOID

class Agent:
    @staticmethod
    def _create_brain(input_size: int, hidden_layer_sizes: list[int], output_size: int, activation_function: ActivationFunction) -> Brain:
        prev_size = input_size
        hidden_layers: list[Layer] = []
        for layer_size in hidden_layer_sizes:
            layer_neurons = []
            for _ in range(layer_size):
                weights = [random.gauss(0, 1) for _ in range(prev_size)]
                bias = random.gauss(0, 1)
                layer_neurons.append(Neuron(weights, bias))
            layer = Layer(layer_neurons, activation_function)
            hidden_layers.append(layer)
            prev_size = layer_size

        output_neurons = []
        for _ in range(output_size):
            weights = [random.gauss(0, 1) for _ in range(prev_size)]
            bias = random.gauss(0, 1)
            output_neurons.append(Neuron(weights, bias))
        output_layer = Layer(output_neurons, activation_function)
        return Brain(input_size, hidden_layers, output_layer)


    def __init__(self, brain: Brain):
        self.brain = brain

    @classmethod
    def new(cls, input_size: int, hidden_layers: list[int], output_size: int, activation_function=SIGMOID):

        return cls(cls._create_brain(input_size, hidden_layers, output_size, activation_function))

    def copy(self):
        return self.__class__(self.brain.copy())

class EvolutionAgent(Agent):
    def learn_random(self, mutation_rate: float = 0.1):
        for layer in self.brain.hiddenLayers:
            for neuron in layer.neurons:
                neuron.weights = [w + random.gauss(0, mutation_rate) for w in neuron.weights]
                neuron.bias += random.gauss(0, mutation_rate)
        for neuron in self.brain.outputLayer.neurons:
            neuron.weights = [w + random.gauss(0, mutation_rate) for w in neuron.weights]
            neuron.bias += random.gauss(0, mutation_rate)

class BackpropagationAgent(Agent):
    def backpropagate(self, input_data: list[float], target_output: list[float], learning_rate: float) -> list[float]:
        self.brain.set_input(input_data)
        self.brain._forward_propagate()

        layers = [*self.brain.hiddenLayers, self.brain.outputLayer]
        error: list[float] = [target_output[i] - neuron.output for i, neuron in enumerate(self.brain.outputLayer.neurons)]

        for layer in reversed(layers):
            error = layer._backpropagate(error, learning_rate)

        return error

        # output_deltas = []
        # old_weights = [n.weights.copy() for n in self.brain.outputLayer.neurons]
        # for i, neuron in enumerate(self.brain.outputLayer.neurons):
        #     error = target_output[i] - neuron.output
        #     delta = error * self.brain.outputLayer.activation_function.slope(neuron.output)
        #     output_deltas.append(delta)
        #     for j in range(len(neuron.weights)):
        #         neuron.weights[j] += learning_rate * delta * self.brain.hiddenLayers[-1].neurons[j].output
        #     neuron.bias += learning_rate * delta

        # current_deltas = output_deltas
        # current_weights = old_weights

        # for i in reversed(range(len(self.brain.hiddenLayers))):
        #     layer = self.brain.hiddenLayers[i]
        #     next_deltas = []
        #     old_weights = [n.weights.copy() for n in layer.neurons]

        #     for j, neuron in enumerate(layer.neurons):
        #         error = sum(current_weights[k][j] * current_deltas[k] for k in range(len(current_deltas)))
        #         delta = error * layer.activation_function.slope(neuron.output)
        #         next_deltas.append(delta)

        #         for k in range(len(neuron.weights)):
        #             if i == 0:
        #                 neuron.weights[k] += learning_rate * delta * self.brain.inputLayer[k]
        #             else:
        #                 neuron.weights[k] += learning_rate * delta * self.brain.hiddenLayers[i - 1].neurons[k].output
        #         neuron.bias += learning_rate * delta

        #     current_deltas = next_deltas
        #     current_weights = old_weights


class ConvolutionalAgent(Agent):
    def __init__(self, brain: ConvolutionalBrain):
        self.brain: ConvolutionalBrain = brain

    def backpropagate(self, input_data: Image, target_output: list[float], learning_rate: float) -> list[list[float]]:
        self.brain.predict_image(input_data)

        layers = [*self.brain.hiddenLayers, self.brain.outputLayer]
        error: list[float] = [target_output[i] - neuron.output for i, neuron in enumerate(self.brain.outputLayer.neurons)]

        for layer in reversed(layers):
            error = layer._backpropagate(error, learning_rate)

        assert self.brain.convs[-1].last_output is not None
        h 
        w = len(self.brain.convs[-1].last_output[0])
        imageError = [error[i:i + w] for i in range(0, len(error), w)]

        for conv in reversed(self.brain.convs):
            imageError = conv._backpropagate(imageError, learning_rate)

        return imageError