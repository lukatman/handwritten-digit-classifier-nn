import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        """ `sizes` contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  #np matrix
        self.weights = [np.random.randn(y, x)                     #np matrix (row: current layer, col: prev.)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def feedforward(self, a):
        #calculate network output from input a (numpy array)
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
            # dot product of weight * a + bias
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # stochastic gradient descent using mini-batches
        # training_data --> (x,y) tuples containing training inputs and desired outputs
        # test_data --> if provided, network will be evaluated against it after each epoch w/ progress print
        # eta --> learning rate
        if test_data:
            test_data = list(test_data) 
            n_test = len(test_data)
        training_data = list(training_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete.")

    def update_mini_batch(self, mini_batch, eta):
        """Apply gradient descent using backpropagation to a single mini-batch 
        to update the network's weights and biases. mini-batch is a list of tuples,
        eta is learning rate"""
        # create new vectors for gradient of bias and weights with same shape but filled with 0s
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # add the gradients of each weight and bias to their corresponding element to get each's total sum
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # subtract (average gradient accross all training examples) from (corresponding weight and bias)
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                    for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
            gradient for the cost function C_x.  "nabla_b" and
            "nabla_w" are layer-by-layer lists of numpy arrays, similar
            to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # store all activations layer by layer
        zs = [] # layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 is the last layer, l = 2 the second-last, and so on
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y) # derivative of the function in textbook

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # derivative of sigmoid
    return sigmoid(z)*(1-sigmoid(z))



