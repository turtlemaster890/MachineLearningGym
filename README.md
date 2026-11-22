Neural Network Agents framework in python

Has both simple weights networks and convolutional networks, both with backpropagation for learning

Has agent abstractions, including an evolutionary agent for things like racing agents

Includes demos for the different kinds of tasks the networks can do

Running:
Download the release exe and run for the demos, or clone locally and run a demo file

Building your own agent:
Make an instance of either Agent (for some pre-built applications) or Brain (for more custom applications) or any of their subclasses
Brain takes a 1d array of float inputs, ConvolutionalBrain takes a 3d array (Images * depth, for something like colour)
ConvolutionalBrain runs a convolutional network in front of its weight arrays, which pass filters over the inputted images which help to recognize features (lines, lights etc)
