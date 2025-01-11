import numpy as np
import random
from enums.activationfunction import ActivationFunctions
from enums.costfunction import CostFunctions
from enums.problemtype import ProblemType
import matplotlib.pyplot as plt
import pandas as pd

class NeuralNetwork():
    def __init__(
            self,
            train_x : np.ndarray,
            train_y : np.ndarray,
            test_x : np.ndarray,
            test_y : np.ndarray,
            hidden_layers : int,
            nb_neurons : int,
            activation_function: ActivationFunctions,
            iteration : int,
            learning_rate : int
            ):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.hidden_layers = hidden_layers
        self.nb_neurons = nb_neurons
        self.activation_function = activation_function
        self.iteration = iteration
        self.learning_rate = learning_rate

        # Initializing caches
        self.W = {}
        self.B = {}
        self.A = {}
        self.Z_cache = {}

        self.output_size = 0
        self.cost_function, self.type = self.select_problemtype()
        self.cost_history = []
    
    def initializing(self):

        """
        Function is used to He initialize weights randomlyand biases and establish the shape of each arrays
        """

        np.random.seed(42)

        for layer in range(self.hidden_layers + 1):
            if layer == 0:
                self.W[layer] = np.random.randn(self.nb_neurons, self.train_x.shape[0]) * np.sqrt(2 / self.train_x.shape[0])
                self.B[layer] = np.random.randn(self.nb_neurons, 1)
            elif layer != self.hidden_layers:
                self.W[layer] = np.random.randn(self.nb_neurons, self.nb_neurons) * np.sqrt(2 / self.train_x.shape[0])
                self.B[layer] = np.random.randn(self.nb_neurons, 1)
            else:
                self.W[layer] = np.random.randn(self.output_size, self.nb_neurons) * np.sqrt(2 / self.train_x.shape[0])
                self.B[layer] = np.random.randn(self.output_size, 1)

    def forward_propagation(self, input : np.ndarray):
        """
        Function is used to perform forward propagation.
        ReLu activation fucntion is used for each layer except output layer

        Args:
            input (np.ndarray): input features to our forward propagation
        """

        for layer in range(self.hidden_layers + 1):
            if layer == 0:
                Z = np.dot(self.W.get(layer), input) + self.B[layer]
                self.Z_cache[layer] = Z
                self.A[layer] = self.relu(Z)

            # Applying activation function of problem type to last layer to get an output
            elif layer == self.hidden_layers:
                Z = np.dot(self.W.get(layer), self.A.get(layer - 1)) + self.B[layer]
                self.Z_cache[layer] = Z
                self.A[layer] = self.activation_func(Z)
                
            else:
                Z = np.dot(self.W.get(layer), self.A.get(layer - 1)) + self.B[layer]
                self.Z_cache[layer] = Z
                self.A[layer] = self.relu(Z)

    def backward_propagation(self):
        """
        Function is used to perform backward propagation
        """        
        dZcache = {}
        dWcache = {}
        dBcache = {}
        
        for layer in range(len(self.A) - 1, -1, -1):

            # Output layer
            if layer == len(self.A) - 1:
                dZ = self.A.get(layer) - self.train_y
                dZcache[layer] = dZ
                dW = 1/self.train_y.shape[1] * np.dot(dZ, np.transpose(self.A.get(layer - 1)))
                dWcache[layer] = dW
                dB = 1/self.train_y.shape[1] * np.sum(dZ, 1).reshape(-1,1)
                dBcache[layer] = dB
            
            # Input Layer
            elif layer == 0:
                dZ = np.dot(np.transpose(self.W.get(layer + 1)), dZcache.get(layer + 1)) * self.d_activation_function(ActivationFunctions.RELU, self.Z_cache.get(layer))
                dZcache[layer] = dZ
                dW = 1/self.train_y.shape[1] * np.dot(dZ, self.train_x.T)
                dWcache[layer] = dW
                dB = 1/self.train_y.shape[1] * np.sum(dZ, 1).reshape(-1,1)
                dBcache[layer] = dB

            # Hidden layers
            else:
                dZ = np.dot(np.transpose(self.W.get(layer + 1)), dZcache.get(layer + 1)) * self.d_activation_function(ActivationFunctions.RELU, self.Z_cache.get(layer))
                dZcache[layer] = dZ
                dW = 1/self.train_y.shape[1] * np.dot(dZ, np.transpose(self.A.get(layer)))
                dWcache[layer] = dW
                dB = 1/self.train_y.shape[1] * np.sum(dZ, 1).reshape(-1,1)
                dBcache[layer] = dB

        # Updating Weights and Biases
        for weights in range(len(self.W)):
            self.W[weights] = self.W[weights] - self.learning_rate * dWcache.get(weights)
        
        for b in range(len(self.B)):
            self.B[b] = self.B[b] - self.learning_rate * dBcache.get(b)        
    
    def d_activation_function(self, act : ActivationFunctions, x : np.ndarray) -> np.ndarray:
        """
        Function is used to output the derivative of the hidden layer activation function (only ReLu for now)

        Args:
            act (ActivationFunctions): Activation Function
            x (np.ndarray): array to compute
        """        
        match act:
            case ActivationFunctions.RELU:
                return x > 0
            
    def get_cost(self) -> float:
        """
        Function to compute cost

        Returns:
            cost: cost of prediction
        """

        pred = self.A.get(self.hidden_layers)
        
        match self.cost_function:
            case CostFunctions.CROSSENTROPY:
                return -(1/self.train_y.shape[1])*np.sum(self.train_y*np.log(pred) + (1-self.train_y) * np.log(1-pred))
            
            case CostFunctions.CLASSCROSSENTROPY:
                return -(1/self.train_y.shape[1])*np.sum(self.train_y*np.log(pred))
            
            case CostFunctions.MSE:
                return (1/(2*self.train_y.shape[1])) * np.sum(pred - self.train_y.T)
        
    def algorithm(self):

        """
        Function to compute full algorithm for a number of iteration
        """
        
        self.initializing()

        for i in range(self.iteration):
            self.forward_propagation(self.train_x)
            cost = self.get_cost()
            self.backward_propagation()
            self.cost_history.append(cost)

            if i % 100 == 0:
                print('Iteration: ', i)
                match self.type:
                    case ProblemType.CLASSIFICATION:
                        print('Accuracy: ', self.get_accuracy(self.get_predictions(self.A.get(self.hidden_layers)), self.train_y))
                    case ProblemType.MULTICLASSIFICATION:
                        print('Accuracy: ', self.get_accuracy(self.get_predictions(self.A.get(self.hidden_layers)), np.argmax(self.train_y, axis=0)))
            
    def relu(self, Z : np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def activation_func(self, x : np.ndarray) -> np.ndarray:
        """
        Function to compute activation function based on problem type

        Args:
            x (np.ndarray): array to compute activation function on

        Returns:
            np.ndarray: activated array
        """        
        match self.activation_function:
            case ActivationFunctions.SIGMOID:
                return 1/(1 + np.exp(-x))
            case ActivationFunctions.SOFTMAX:
                return np.exp(x)/np.sum(np.exp(x), axis = 0)
            case _:
                return x # No need for an activation function in regression problems
    
    def select_problemtype(self):
        """
        Function that classifies cost function and problem type based on input of activation function from user

        Returns:
            CostFunctions, ProblemType: returns costfunction and problem type
        """        
        match self.activation_function:
            case ActivationFunctions.SIGMOID:
                self.output_size = 1
                return CostFunctions.CROSSENTROPY, ProblemType.CLASSIFICATION
            case ActivationFunctions.SOFTMAX:
                self.output_size = self.test_y.shape[0]
                return CostFunctions.CLASSCROSSENTROPY, ProblemType.MULTICLASSIFICATION
            case _:
                self.output_size = 1
                return CostFunctions.MSE, ProblemType.REGRESSION
    
    def predict(self, features : np.ndarray) -> np.ndarray:
        """
        Function to make a new prediction

        Args:
            features (np.ndarray): takes new features input 

        Returns:
            np.ndarray: returns the output of neural network
        """        
        self.forward_propagation(features)
        return self.A.get(self.hidden_layers)
    
    def get_predictions(self, A : np.ndarray) -> np.ndarray:
        """
        Function to modify output based on problem type

        Args:
            A (np.ndarray): last activation layer

        Returns:
            np.ndarray: modified output
        """        
        if self.type == ProblemType.MULTICLASSIFICATION:
            return np.argmax(A,0)
        elif self.type == ProblemType.CLASSIFICATION:
            return np.where(np.array(A) >= 0.5, 1, 0).reshape(-1)
        else:
            pass
        
    def get_accuracy(self, predictions : np.ndarray, Y : np.ndarray) -> float:
        """
        Function to get the accuracy of prediction

        Args:
            predictions (np.ndarray): array of predictions made by model
            Y (np.ndarray): array of actual values 

        Returns:
            float: returns proportion of correct values
        """        
        return np.sum(predictions == Y)/ Y.size
    
    def plot_error_function(self):
        """
        This function is used to plot the error function
        """
        plt.plot(range(self.iteration), self.cost_history, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function during gradient descent')
        plt.show()

    def test_pred(self):
        """
        Function to apply test set and get info on performance
        """
        
        prediction = self.get_predictions(self.predict(self.test_x))

        match self.type:
            case ProblemType.MULTICLASSIFICATION:
                actual = np.argmax(self.test_y, axis=0)
            case ProblemType.CLASSIFICATION:
                actual = self.test_y.reshape(-1)

        df = pd.DataFrame({
            'prediction' : prediction,
            'actual' : actual
        })
        print('Testing set:')
        print('Accuracy: ', self.get_accuracy(prediction, actual))
        print('\n')
        print('Testing matrix of predicted valued vs actual values')
        pivot = pd.crosstab (df['prediction'], df['actual'])
        print(pivot)