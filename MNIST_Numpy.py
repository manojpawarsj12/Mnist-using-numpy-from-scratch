from __future__ import print_function
from IPython.display import clear_output
# Import Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
np.random.seed(42)
import mnist

class Layer:
    def __init__(self):
     
        self.weights = np.zeros(shape=(input.shape[1], 10))
        bias = np.zeros(shape=(10,))
        pass

    def forward(self, input):
        
        output = np.matmul(input, self.weights) + bias
        return output


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate

    
        self.weights = np.random.randn(input_units, output_units)*0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        return np.matmul(input, self.weights) + self.biases

    def backward(self, input, grad_output):
       
        grad_input = np.dot(grad_output, np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output), input))
        grad_biases = np.sum(grad_output, axis=0)

       
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input


class ReLU(Layer):
    def __init__(self):
  
        pass

    def forward(self, input):
     
        return np.maximum(0, input)

    def backward(self, input, grad_output):
      
        relu_grad = input > 0
        return grad_output*relu_grad


def softmax_crossentropy_with_logits(logits, reference_answers):
    
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
 
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]


def load_dataset(flatten=False):
   # (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = mnist.train_images()
    y_train = mnist.train_labels()

    X_test = mnist.test_images()
    y_test = mnist.test_labels()
   
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

plt.figure(figsize=[6, 6])
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.title("Label: %i" % y_train[i])
    plt.imshow(X_train[i].reshape([28, 28]), cmap='gray');

network = []
network.append(Dense(X_train.shape[1], 100))
network.append(ReLU())
network.append(Dense(100, 200))
network.append(ReLU())
network.append(Dense(200, 10))


def forward(network, X):

    activations = []
    input = X
    for i in range(len(network)):
        activations.append(network[i].forward(X))
        X = network[i].forward(X)

    assert len(activations) == len(network)
    return activations


def predict(network, X):
 
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
 

    # Get the layer activations
    layer_activations = forward(network, X)
    logits = layer_activations[-1]

    # Compute the loss and the initial gradient
    loss = softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

    for i in range(1, len(network)):
        loss_grad = network[len(
            network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)

    return np.mean(loss)

  
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
train_log = []
val_log = []
for epoch in range(25):

    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=32,shuffle=True):
        train(network,x_batch,y_batch)
    
    train_log.append(np.mean(predict(network,X_train)==y_train))
    val_log.append(np.mean(predict(network,X_val)==y_val))
    
    clear_output()
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])
    plt.plot(train_log,label='train accuracy')
    plt.plot(val_log,label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
