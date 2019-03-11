import pynet
import random


# function we want to approximate
def xor(a, b):
    return (a + b) % 2


# list of possible input pairs for xor
cases = [[0, 0], [0, 1], [1, 0], [1, 1]]

# give input count, hidden node count, output count and "learn rate"
nn = pynet.NeuralNetwork(2, 2, 1, 0.1)

# train on known inputs
for i in range(50000):
    inputs = random.choice(cases)
    target = xor(*inputs)
    nn.train(inputs, [target])  # output is returned here, if you want it

# can now freely classify "unknown" examples
for inputs in cases:
    output = nn.assess(inputs)[0]  # index 0 to unpack size-1 list
    guess = round(output)  # output is a float, we want 0 or 1
    if guess == 0:  # certainty if based on the value of the float
        certainty = 100 - round(output*100)
    else:
        certainty = round(output*100)

    print(f"{inputs} -> {guess} ({certainty}% sure)")
