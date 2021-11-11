import numpy as np
from layer import Dense
import nnfs
from nnfs.datasets import spiral_data
from activation import ActivationReLU, ActivationSoftMax, LossCategoricalCrossEntropy

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 3)
activation1 = ActivationReLU()

dense2 = Dense(3, 3)
activation2 = ActivationSoftMax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = LossCategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)
