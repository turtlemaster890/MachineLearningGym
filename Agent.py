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
    ) -> ConvolutionalBrain:
        prev_image_size = input_image_size
        conv_layers: list[ConvolutionalLayer] = []
        prev_channels = input_channels
        for filter_size in conv_sizes:
            filter_width, filter_height = filter_size
            filter_weights: list[list[list[float]]] = [
                [[0] * filter_width for _ in range(filter_height)]
                for _ in range(prev_channels)
            ]
            for c in range(prev_channels):
                for w in range(filter_width):
                    for h in range(filter_height):
                        filter_weights[c][h][w] = random.gauss(0, 1)
            bias = random.gauss(0, 1)
            filter = Filter(filter_weights, bias)
            conv_layer = ConvolutionalLayer([filter], conv_activation)
            conv_layers.append(conv_layer)
            prev_image_size = (
                prev_image_size[0] - filter_height + 1,
                prev_image_size[1] - filter_width + 1,
            )
            prev_channels = len(conv_layer.filters)

        prev_size = prev_channels * prev_image_size[0] * prev_image_size[1]
        input_size = prev_size
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
