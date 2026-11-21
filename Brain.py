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

type Error = Image


class ConvolutionalLayer:
    def __init__(
        self,
        filter: list[list[float]],
        bias: float,
        activation_function: ActivationFunction = RELU,
    ):
        self.filter = filter
        self.bias = bias
        self.activation_function = activation_function

        self.last_image: Image | None = None
        self.last_output: Image | None = None

    def apply(self, image: list[list[float]]) -> list[list[float]]:
        self.last_image = image
        fH, fW = len(self.filter), len(self.filter[0])
        iH, iW = len(image), len(image[0])
        outW, outH = iW - fW + 1, iH - fH + 1
        out: list[list[float]] = [[0] * (outW) for _ in range(outH)]
        self.last_output = [[0] * (outW) for _ in range(outH)]
        for i in range(outH):
            for j in range(outW):
                total = self.bias
                for k in range(fH):
                    for l in range(fW):
                        total += image[i + k][j + l] * self.filter[k][l]
                val = self.activation_function.func(total)
                out[i][j] = val
                self.last_output[i][j] = val
        return out

    def _backpropagate(self, error: Error, learning_rate: float) -> Error:
        assert self.last_image is not None
        assert self.last_output is not None
        eH, eW = len(error), len(error[0])
        fH, fW = len(self.filter), len(self.filter[0])
        inputError: Image = [[0] * (eW + fW - 1) for _ in range(eH + fH - 1)]
        filterNudge: Error = [[0] * fW for _ in range(fH)]
        biasNudge = 0
        for i in range(eH):
            for j in range(eW):
                e = error[i][j] * self.activation_function.slope(self.last_output[i][j])
                for k in range(fH):
                    for l in range(fW):
                        filterNudge[k][l] += e * self.last_image[i + k][j + l]
                        inputError[i + k][j + l] += e * self.filter[k][l]
                biasNudge += e
        for k in range(fH):
            for l in range(fW):
                self.filter[k][l] += learning_rate * filterNudge[k][l]
        self.bias += learning_rate * biasNudge

        return inputError

    def copy(self) -> ConvolutionalLayer:
        copied_layer = ConvolutionalLayer(
            [row.copy() for row in self.filter], self.bias, self.activation_function
        )
        return copied_layer


class ConvolutionalLayerDict(TypedDict):
    filter: list[list[float]]
    bias: float


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


class ConvolutionalBrainDict(BrainDict):
    convLayers: list[ConvolutionalLayerDict]


class ConvolutionalBrain(Brain):
    def __init__(self, convs: list[ConvolutionalLayer], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convs = convs
        self.image: list[list[float]] = []

    def _forward_propagate(self):
        x = self.image
        for c in self.convs:
            x = c.apply(x)
        self.inputLayer = [pixel for row in x for pixel in row]
        super()._forward_propagate()

    def predict_image(self, image: list[list[float]]) -> list[float]:
        self.image = image
        self._forward_propagate()
        return self.get_output()

    def _save(self) -> ConvolutionalBrainDict:
        s = super()._save()
        return {
            **s,
            "convLayers": [{"filter": c.filter, "bias": c.bias} for c in self.convs],
        }

    @classmethod
    def _load(cls, data: ConvolutionalBrainDict, activation_function: ActivationFunction = SIGMOID) -> Self:  # type: ignore
        conv_layers = [
            ConvolutionalLayer(c["filter"], c["bias"], activation_function)
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
