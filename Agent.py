import math
import random

from Brain import (
    SIGMOID,
    ActivationFunction,
    Brain,
    ConvolutionalBrain,
    ConvolutionalLayer,
    Filter,
    Layer,
    Neuron,
    Volume,
)


class Agent:
    def __init__(self, brain: Brain):
        self.brain = brain


class LayeredAgent(Agent):
    @staticmethod
    def _create_brain(
        input_size: int,
        hidden_layer_sizes: list[int],
        output_size: int,
        hidden_layer_activation: ActivationFunction = SIGMOID,
        output_layer_activation: ActivationFunction = SIGMOID,
    ) -> Brain:
        prev_size = input_size
        hidden_layers: list[Layer] = []
        for layer_size in hidden_layer_sizes:
            layer_neurons = []
            for _ in range(layer_size):
                weights = [random.gauss(0, 1) for _ in range(prev_size)]
                bias = random.gauss(0, 1)
                layer_neurons.append(Neuron(weights, bias))
            layer = Layer(layer_neurons, hidden_layer_activation)
            hidden_layers.append(layer)
            prev_size = layer_size

        output_neurons = []
        for _ in range(output_size):
            weights = [random.gauss(0, 1) for _ in range(prev_size)]
            bias = random.gauss(0, 1)
            output_neurons.append(Neuron(weights, bias))
        output_layer = Layer(output_neurons, output_layer_activation)
        return Brain(input_size, hidden_layers, output_layer)

    @classmethod
    def new(cls, input_size: int, hidden_layers: list[int], output_size: int):

        return cls(cls._create_brain(input_size, hidden_layers, output_size))

    def copy(self):
        return self.__class__(self.brain.copy())


class EvolutionAgent(LayeredAgent):
    def learn_random(self, mutation_rate: float = 0.1):
        for layer in self.brain.hiddenLayers:
            for neuron in layer.neurons:
                neuron.weights = [
                    w + random.gauss(0, mutation_rate) for w in neuron.weights
                ]
                neuron.bias += random.gauss(0, mutation_rate)
        for neuron in self.brain.outputLayer.neurons:
            neuron.weights = [
                w + random.gauss(0, mutation_rate) for w in neuron.weights
            ]
            neuron.bias += random.gauss(0, mutation_rate)


class BackpropagationAgent(LayeredAgent):
    def backpropagate(
        self, input_data: list[float], target_output: list[float], learning_rate: float
    ) -> list[float]:
        self.brain.set_input(input_data)
        self.brain._forward_propagate()

        layers = [*self.brain.hiddenLayers, self.brain.outputLayer]
        error: list[float] = [
            target_output[i] - neuron.output
            for i, neuron in enumerate(self.brain.outputLayer.neurons)
        ]

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

    @staticmethod
    def _create_convolutional_brain(
        input_image_size: tuple[int, int],
        conv_sizes: list[tuple[int, int]],
        hidden_layer_sizes: list[int],
        output_size: int,
        conv_activation: ActivationFunction = SIGMOID,
        hidden_layer_activation: ActivationFunction = SIGMOID,
        output_layer_activation: ActivationFunction = SIGMOID,
        input_channels: int = 3,
        filters_per_layer: int | list[int] = 1,
    ) -> ConvolutionalBrain:
        if isinstance(filters_per_layer, int):
            filters_per_layer = [filters_per_layer] * len(conv_sizes)

        def dense_layer(n_in: int, n_out: int, activation: ActivationFunction) -> Layer:
            scale = 1 / math.sqrt(n_in)
            return Layer(
                [
                    Neuron([random.gauss(0, scale) for _ in range(n_in)], 0.0)
                    for _ in range(n_out)
                ],
                activation,
            )

        prev_image_size = input_image_size
        prev_channels = input_channels
        conv_layers: list[ConvolutionalLayer] = []

        for (filter_width, filter_height), n_filters in zip(
            conv_sizes, filters_per_layer
        ):
            scale = 1 / math.sqrt(prev_channels * filter_width * filter_height)
            filters = [
                Filter(
                    [
                        [
                            [random.gauss(0, scale) for _ in range(filter_width)]
                            for _ in range(filter_height)
                        ]
                        for _ in range(prev_channels)
                    ],
                    0.0,
                )
                for _ in range(n_filters)
            ]
            conv_layers.append(ConvolutionalLayer(filters, conv_activation))
            prev_image_size = (
                prev_image_size[0] - filter_height + 1,
                prev_image_size[1] - filter_width + 1,
            )
            prev_channels = n_filters

        prev_size = prev_channels * prev_image_size[0] * prev_image_size[1]
        input_size = prev_size

        hidden_layers: list[Layer] = []
        for layer_size in hidden_layer_sizes:
            hidden_layers.append(
                dense_layer(prev_size, layer_size, hidden_layer_activation)
            )
            prev_size = layer_size

        output_layer = dense_layer(prev_size, output_size, output_layer_activation)

        return ConvolutionalBrain(
            conv_layers,
            input_size,
            hiddenLayers=hidden_layers,
            outputLayer=output_layer,
        )

    @classmethod
    def new(
        cls,
        input_image_size: tuple[int, int],
        conv_sizes: list[tuple[int, int]],
        hidden_layer_sizes: list[int],
        output_size: int,
        conv_activation: ActivationFunction = SIGMOID,
        hidden_layer_activation: ActivationFunction = SIGMOID,
        output_layer_activation: ActivationFunction = SIGMOID,
        input_channels: int = 3,
        filters_per_layer: int | list[int] = 1,
    ):
        return cls(
            cls._create_convolutional_brain(
                input_image_size,
                conv_sizes,
                hidden_layer_sizes,
                output_size,
                conv_activation,
                hidden_layer_activation,
                output_layer_activation,
                input_channels,
                filters_per_layer,
            )
        )

    def backpropagate(
        self, input_data: Volume, target_output: list[float], learning_rate: float
    ) -> Volume:
        self.brain.predict_image(input_data)

        layers = [*self.brain.hiddenLayers, self.brain.outputLayer]
        error: list[float] = [
            target_output[i] - neuron.output
            for i, neuron in enumerate(self.brain.outputLayer.neurons)
        ]

        for layer in reversed(layers):
            error = layer._backpropagate(error, learning_rate)

        assert self.brain.convs[-1].last_output is not None
        c = len(self.brain.convs[-1].last_output)
        h = len(self.brain.convs[-1].last_output[0])
        w = len(self.brain.convs[-1].last_output[0][0])
        plane_size = h * w
        imageError = [
            [
                error[ch * plane_size + r * w : ch * plane_size + (r + 1) * w]
                for r in range(h)
            ]
            for ch in range(c)
        ]

        for conv in reversed(self.brain.convs):
            imageError = conv._backpropagate(imageError, learning_rate)

        return imageError
