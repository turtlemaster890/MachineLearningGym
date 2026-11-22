from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, TypedDict


@dataclass
class ActivationFunction:
    func: Callable[[float], float]
    slope: Callable[[float], float]


SIGMOID = ActivationFunction(
    func=lambda x: 1 / (1 + pow(math.e, -x)), slope=lambda x: x * (1 - x)
)

RELU = ActivationFunction(func=lambda x: max(0, x), slope=lambda x: 1 if x > 0 else 0)


@dataclass
class Neuron:
    weights: list[float]
    bias: float
    output: float = 0

    def copy(self):
        return Neuron(self.weights.copy(), self.bias, self.output)

    @classmethod
    def from_dict(cls, data: NeuronDict) -> Neuron:
        return cls(data["weights"], data["bias"], data["output"])


class NeuronDict(TypedDict):
    weights: list[float]
    bias: float
    output: float


class BrainDict(TypedDict):
    inputSize: int
    hiddenLayers: list[list[NeuronDict]]
    outputLayer: list[NeuronDict]


class Layer:
    def __init__(
        self, neurons: list[Neuron], activation_function: ActivationFunction = SIGMOID
    ):
        self.neurons = neurons
        self.activation_function = activation_function

        self.last_inputs: list[float] | None = None

    def apply(self, inputs: list[float]) -> list[float]:
        outputs = []
        self.last_inputs = inputs
        for neuron in self.neurons:
            neuron_input = (
                sum(w * inp for w, inp in zip(neuron.weights, inputs)) + neuron.bias
            )
            neuron.output = self.activation_function.func(neuron_input)
            outputs.append(neuron.output)
        return outputs

    def _backpropagate(self, errors: list[float], learning_rate: float) -> list[float]:
        assert self.last_inputs is not None
        input_errors: list[float] = [0] * len(self.neurons[0].weights)
        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * self.activation_function.slope(neuron.output)
            for j in range(len(neuron.weights)):
                input_errors[j] += delta * neuron.weights[j]
                neuron.weights[j] += learning_rate * delta * self.last_inputs[j]
            neuron.bias += learning_rate * delta
        return input_errors

    def copy(self) -> Layer:
        copied_neurons = [neuron.copy() for neuron in self.neurons]
        return Layer(copied_neurons, self.activation_function)


type Image = list[list[float]]

type Volume = list[Image]

type Error = Image


@dataclass
class Filter:
    weights: Volume
    bias: float

    def apply_to_cell(self, volume: Volume, i: int, j: int) -> float:
        C = len(self.weights)
        fH, fW = len(self.weights[0]), len(self.weights[0][0])
        total = self.bias
        for c in range(C):
            for k in range(fH):
                for l in range(fW):
                    total += self.weights[c][k][l] * volume[c][i + k][j + l]
        return total

    def copy(self) -> Filter:
        copied_weights = [[row.copy() for row in channel] for channel in self.weights]
        return Filter(copied_weights, self.bias)


class ConvolutionalLayer:
    def __init__(
        self,
        filters: list[Filter],
        activation_function: ActivationFunction = RELU,
    ):
        self.filters = filters
        self.activation_function = activation_function

        self.last_input: Volume | None = None
        self.last_output: Volume | None = None

    def apply(self, volume: Volume) -> Volume:
        self.last_input = volume
        C, iH, iW = len(volume), len(volume[0]), len(volume[0][0])
        fH, fW = len(self.filters[0].weights[0]), len(self.filters[0].weights[0][0])
        outH, outW = iH - fH + 1, iW - fW + 1

        out: Volume = []

        for filter in self.filters:
            feat_map: Image = [[0] * outW for _ in range(outH)]
            for i in range(outH):
                for j in range(outW):
                    feat_map[i][j] = self.activation_function.func(
                        filter.apply_to_cell(volume, i, j)
                    )
            out.append(feat_map)

        self.last_output = out
        return out

    def _backpropagate(self, error: Volume, learning_rate: float) -> Volume:
        assert self.last_input is not None
        assert self.last_output is not None
        oH, oW = len(self.last_output[0]), len(self.last_output[0][0])
        # fH, fW = len(self.filters[0].weights), len(self.filters[0].weights[0])
        inputError: Volume = [
            [[0] * len(self.last_input[0][0]) for _ in range(len(self.last_input[0]))]
            for _ in range(len(self.last_input))
        ]
        filterNudges: list[Volume] = [
            [
                [[0] * len(f.weights[0][0]) for _ in range(len(f.weights[0]))]
                for _ in range(len(f.weights))
            ]
            for f in self.filters
        ]
        biasNudges: list[float] = [0] * len(self.filters)
        for i in range(oH):
            for j in range(oW):
                for idx, filter in enumerate(self.filters):
                    delta = error[idx][i][j] * self.activation_function.slope(
                        self.last_output[idx][i][j]
                    )
                    biasNudges[idx] += delta
                    for c in range(len(filter.weights)):
                        for k in range(len(filter.weights[0])):
                            for l in range(len(filter.weights[0][0])):
                                filterNudges[idx][c][k][l] += (
                                    delta * self.last_input[c][i + k][j + l]
                                )
                                inputError[c][i + k][j + l] += (
                                    delta * filter.weights[c][k][l]
                                )
        for idx, filter in enumerate(self.filters):
            for c in range(len(filter.weights)):
                for k in range(len(filter.weights[0])):
                    for l in range(len(filter.weights[0][0])):
                        filter.weights[c][k][l] += (
                            learning_rate * filterNudges[idx][c][k][l]
                        )
        for idx, filter in enumerate(self.filters):
            filter.bias += learning_rate * biasNudges[idx]

        return inputError

    def copy(self) -> ConvolutionalLayer:
        copied_filters = [
            f.copy() for f in self.filters
        ]
        copied_layer = ConvolutionalLayer(copied_filters, self.activation_function)
        return copied_layer


class Brain:
    def __init__(
        self,
        inputSize: int,
        hiddenLayers: list[Layer],
        outputLayer: Layer,
        # activation_function: ActivationFunction = SIGMOID,
    ):
        self.inputLayer: list[float] = [0] * inputSize
        self.hiddenLayers: list[Layer] = hiddenLayers
        self.outputLayer: Layer = outputLayer
        # self._activation_function = activation_function

    def copy(self):
        # copied_hidden_layers = [
        #     [neuron.copy() for neuron in layer.neurons] for layer in self.hiddenLayers
        # ]
        copied_hidden_layers = [layer.copy() for layer in self.hiddenLayers]
        return self.__class__(
            len(self.inputLayer),
            copied_hidden_layers,
            self.outputLayer.copy(),
            # self._activation_function,
        )

    def set_input(self, inputData: list[float]):
        if len(inputData) != len(self.inputLayer):
            raise ValueError("Input data size does not match input layer size.")

        self.inputLayer = inputData

    def get_output(self) -> list[float]:
        return [neuron.output for neuron in self.outputLayer.neurons]

    def _forward_propagate(self):
        inputs = self.inputLayer
        for i, layer in enumerate(self.hiddenLayers):
            # for j, neuron in enumerate(layer):
            #     if i == 0:
            #         neuron_input = (
            #             sum(w * inp for w, inp in zip(neuron.weights, self.inputLayer))
            #             + neuron.bias
            #         )
            #     else:
            #         neuron_input = (
            #             sum(
            #                 w * self.hiddenLayers[i - 1][k].output
            #                 for k, w in enumerate(neuron.weights)
            #             )
            #             + neuron.bias
            #         )
            #     neuron.output = self._activation_function.func(neuron_input)
            inputs = layer.apply(inputs)

        self.outputLayer.apply(inputs)
        # for j, neuron in enumerate(self.outputLayer):
        # neuron_input = (
        #     sum(
        #         w * self.hiddenLayers[-1][k].output
        #         for k, w in enumerate(neuron.weights)
        #     )
        #     + neuron.bias
        # )
        # neuron.output = self._activation_function.func(neuron_input)

    def predict(self, inputData: list[float]) -> list[float]:
        self.set_input(inputData)
        self._forward_propagate()
        return self.get_output()

    def _save(self) -> BrainDict:
        return {
            "inputSize": len(self.inputLayer),
            "hiddenLayers": [
                [
                    {
                        "weights": neuron.weights,
                        "bias": neuron.bias,
                        "output": neuron.output,
                    }
                    for neuron in layer.neurons
                ]
                for layer in self.hiddenLayers
            ],
            "outputLayer": [
                {
                    "weights": neuron.weights,
                    "bias": neuron.bias,
                    "output": neuron.output,
                }
                for neuron in self.outputLayer.neurons
            ],
        }

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self._save(), f, indent=4)

    @classmethod
    def _load(
        cls, data: BrainDict, activation_function: ActivationFunction = SIGMOID
    ) -> Self:
        hidden_layers = [
            Layer(
                [Neuron.from_dict(neuron_data) for neuron_data in layer_data],
                activation_function,
            )
            for layer_data in data["hiddenLayers"]
        ]
        output_layer = Layer(
            [Neuron.from_dict(neuron_data) for neuron_data in data["outputLayer"]],
            activation_function,
        )
        return cls(data["inputSize"], hidden_layers, output_layer)

    @classmethod
    def load(
        cls, filename: str, activation_function: ActivationFunction = SIGMOID
    ) -> Brain:
        with open(filename, "r") as f:
            data = json.load(f)
        return cls._load(data, activation_function)


class ConvolutionalLayerDict(TypedDict):
    filter: Volume
    bias: float


class ConvolutionalBrainDict(BrainDict):
    convLayers: list[list[ConvolutionalLayerDict]]


class ConvolutionalBrain(Brain):
    def __init__(self, convs: list[ConvolutionalLayer], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convs = convs
        self.volume: Volume = []

    def _forward_propagate(self):
        x = self.volume
        for c in self.convs:
            x = c.apply(x)
        self.inputLayer = [pixel for channel in x for row in channel for pixel in row]
        super()._forward_propagate()

    def predict_image(self, volume: Volume) -> list[float]:
        self.volume = volume
        self._forward_propagate()
        return self.get_output()

    def _save(self) -> ConvolutionalBrainDict:
        s = super()._save()
        return {
            **s,
            "convLayers": [
                [{"filter": f.weights, "bias": f.bias} for f in c.filters]
                for c in self.convs
            ],
        }

    @classmethod
    def _load(cls, data: ConvolutionalBrainDict, activation_function: ActivationFunction = SIGMOID) -> Self:  # type: ignore
        conv_layers = [
            ConvolutionalLayer(
                [Filter(f["filter"], f["bias"]) for f in c], activation_function
            )
            for c in data["convLayers"]
        ]
        return cls(
            conv_layers,
            data["inputSize"],
            [
                Layer(
                    [Neuron.from_dict(neuron_data) for neuron_data in layer_data],
                    # activation_function,
                )
                for layer_data in data["hiddenLayers"]
            ],
            Layer(
                [Neuron.from_dict(neuron_data) for neuron_data in data["outputLayer"]],
                # activation_function,
            ),
        )
        # brain = super()._load(data, activation_function)
        # brain.convs = conv_layers
        # return brain

    def copy(self):
        copied_convs = [c.copy() for c in self.convs]
        return ConvolutionalBrain(
            copied_convs,
            len(self.inputLayer),
            [layer.copy() for layer in self.hiddenLayers],
            self.outputLayer.copy(),
            # self._activation_function,
        )
