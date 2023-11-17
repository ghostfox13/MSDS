import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases for the hidden layer and output layer
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def mse_loss(self, predicted, actual):
        return np.mean(np.square(predicted - actual))
    
    def binary_crossentropy_loss(self, predicted, actual):
        epsilon = 1e-15
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
    
    def forward_propagation(self, inputs):
        # Hidden layer calculation
        self.hidden_layer_activation = np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_activation)  # You can switch to sigmoid if you prefer
        
        # Output layer calculation
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        self.predicted_output = self.sigmoid(self.output_layer_activation)
        
        return self.predicted_output
    
    def backward_propagation(self, inputs, targets):
        # Calculate output layer error and delta
        output_error = targets - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)
        
        # Calculate hidden layer error and delta
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_layer_output)  # Change to sigmoid derivative if using sigmoid
        
        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * self.learning_rate
        self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * self.learning_rate
        self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            # Forward propagation
            predicted_output = self.forward_propagation(inputs)
            
            # Backward propagation
            self.backward_propagation(inputs, targets)
            
            # Calculate loss
            loss = self.binary_crossentropy_loss(predicted_output, targets)  # Change the loss function if needed
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss}")
    
    def predict(self, inputs):
        return self.forward_propagation(inputs)
# Define the XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize the neural network with input size 2, one hidden layer with 4 neurons, and output size 1
model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

# Train the model
model.train(X, y, epochs=10000)

# Example: Changing the number of neurons in the hidden layer and using different activation functions
model = NeuralNetwork(input_size=2, hidden_size=8, output_size=1, learning_rate=0.01)
model.train(X, y, epochs=5000)
