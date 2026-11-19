from __future__ import annotations

import math
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict


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

def sigmoid(x: float) -> float:
    return 1 / (1 + pow(math.e, -x))

def relu(x: float) -> float:
    return max(0, x)

class Brain:
    def __init__(self, inputSize: int, hiddenLayers: list[list[Neuron]], outputLayer: list[Neuron], activation_function: Callable[[float], float] = sigmoid):
        self.inputLayer: list[float] = [0] * inputSize
        self.hiddenLayers: list[list[Neuron]] = hiddenLayers
        self.outputLayer: list[Neuron] = outputLayer
        self._activation_function = activation_function

    def copy(self):
        copied_hidden_layers = [[neuron.copy() for neuron in layer] for layer in self.hiddenLayers]
        copied_output_layer = [neuron.copy() for neuron in self.outputLayer]
        return Brain(len(self.inputLayer), copied_hidden_layers, copied_output_layer, self._activation_function)

    def set_input(self, inputData: list[float]):
        if len(inputData) != len(self.inputLayer):
            raise ValueError("Input data size does not match input layer size.")

        self.inputLayer = inputData

    def get_output(self) -> list[float]:
        return [neuron.output for neuron in self.outputLayer]

    def _forward_propagate(self):
        for i, layer in enumerate(self.hiddenLayers):
            for j, neuron in enumerate(layer):
                if i == 0:
                    neuron_input = sum(w * inp for w, inp in zip(neuron.weights, self.inputLayer)) + neuron.bias
                else:
                    neuron_input = sum(w * self.hiddenLayers[i - 1][k].output for k, w in enumerate(neuron.weights)) + neuron.bias
                neuron.output = self._activation_function(neuron_input)

        for j, neuron in enumerate(self.outputLayer):
            neuron_input = sum(w * self.hiddenLayers[-1][k].output for k, w in enumerate(neuron.weights)) + neuron.bias
            neuron.output = self._activation_function(neuron_input)

    def predict(self, inputData: list[float]) -> list[float]:
        self.set_input(inputData)
        self._forward_propagate()
        return self.get_output()

    def _save(self) -> BrainDict:
        return {
            "inputSize": len(self.inputLayer),
            "hiddenLayers": [
                [
                    {"weights": neuron.weights, "bias": neuron.bias, "output": neuron.output}
                    for neuron in layer
                ]
                for layer in self.hiddenLayers
            ],
            "outputLayer": [
                {"weights": neuron.weights, "bias": neuron.bias, "output": neuron.output}
                for neuron in self.outputLayer
            ]
        }

    def save(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self._save(), f, indent=4)

    @classmethod
    def _load(cls, data: BrainDict, activation_function: Callable[[float], float] = sigmoid) -> Brain:
        hidden_layers = [
            [Neuron.from_dict(neuron_data) for neuron_data in layer_data]
            for layer_data in data["hiddenLayers"]
        ]
        output_layer = [Neuron.from_dict(neuron_data) for neuron_data in data["outputLayer"]]
        return cls(data["inputSize"], hidden_layers, output_layer, activation_function)

    @classmethod
    def load(cls, filename: str, activation_function: Callable[[float], float] = sigmoid) -> Brain:
        with open(filename, "r") as f:
            data = json.load(f)
        return cls._load(data, activation_function)