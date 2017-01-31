from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator to get same values always
        random.seed(1)

        # Model a single neuron with 3 input connections and 1 output connection
        # We assign random weights to a 3 x 1 matrix, with values in the range
        # -1 to 1 and mean 0
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    # The sigmoid function, which describes an s shaped curve
    # We pass the weighted sum of the inputs through this function
    # to normalize them between 0 and 1
    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    # Gradient of the sigmoid function
    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, num_iter):
        for i in range(num_iter):
            # Pass the training set through our neural net
            output = self.predict(training_set_inputs)

            # Calculate the error
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the
            # sigmoid curve
            adjustment = dot(training_set_inputs.T,
                             error * self.__sigmoid_derivative(output))

            # Adjust the weights
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        # Pass inputs through our neural network of single neuron
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':

    # initialize a single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: {}"
           .format(neural_network.synaptic_weights))

    # The training set.
    # Four examples, each consisting of 3 input and 1 output value
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each times
    num_iterations = 10000
    neural_network.train(training_set_inputs, training_set_outputs, num_iterations)

    print("New synaptic weights after training: {}"
           .format(neural_network.synaptic_weights))

    # Test the neural network
    print("Predicting: {}"
           .format(neural_network.predict(array([1,0,0]))))
