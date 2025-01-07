import numpy as np
import random
from enums.activationfunction import ActivationFunctions
from enums.costfunction import CostFunctions
from enums.problemtype import ProblemType
import matplotlib.pyplot as plt

class NeuralNetwork():
    # TODO Add activation_function other than sigmoid
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

        self.W = {}
        self.B = np.array([0 for _ in range(hidden_layers)])
        self.A = {}
        self.cost_function = self.select_costfunction()
        self.cost_history = []
        self.type = self.type_problem()

        if (self.activation_function != ActivationFunctions.SIGMOID) & \
           ((self.train_y == 0) | (self.train_y == 1)).all():
            raise ValueError(f'Binary target value necessitate activation function: {self.activation_function} ')
        
    def type_problem(self):
        match self.cost_function:
            case CostFunctions.CROSSENTROPY:
                return ProblemType.CLASSIFICATION
            case CostFunctions.CLASSCROSSENTROPY:
                return ProblemType.MULTICLASSIFICATION
            case CostFunctions.MSE:
                return ProblemType.REGRESSION
            
    def initializing(self):
        random.seed(42)
        for layer in range(self.hidden_layers):
            if layer == 0:
                self.W[layer] = np.random.randn(self.nb_neurons, self.train_x.shape[1])
            else:
                self.W[layer] = np.random.randn(self.nb_neurons, self.train_x.shape[1])

        
    def forward_propagation(self, input : np.ndarray):
        for layer in range(self.hidden_layers):
            if layer == 0:
                Z = np.dot(self.W.get(layer), input) + self.B[layer]
                self.A[layer] = self.activation_function(Z)
            else:
                Z = np.dot(self.W.get(layer), self.A.get(layer - 1)) + self.B[layer]
                self.A[layer] = self.activation_function(Z)

    def backward_propagation(self):
        dZcache = {}
        dWcache = {}
        dBcache = {}
        
        for layer in range(len(self.A) - 1, -1, -1):
            if layer == len(self.A):
                dZlast = self.A.get(layer) - self.test_y
                dZcache[layer] = dZlast
                dWlast = 1/self.train_x.shape[0] * dZlast * np.transpose(self.A.get(layer - 1))
                dWcache[layer] = dWlast
                dBlast = 1/self.train_x.shape[0] * np.sum(dZlast, 1)
                dBcache[layer] = dBlast
            else:
                Z = np.dot(self.W.get(layer), self.A.get(layer - 1)) + self.B[layer]
                dZ = self.W.get(layer) * dZcache.get(layer + 1) * self.d_activation_function(Z)
                dZcache[layer] = dZ
                dW = 1/self.test_x.shape[0] * dZ * np.transpose(self.A.get(layer))
                dWcache = dW
                dB = 1/self.test_x.shape[0] * np.sum(dZ, 1)
                dBcache = dB

        for weights in range(len(self.W)):
            self.W[weights] = self.W.get(weights) - self.learning_rate * dWcache.get(weights)
        
        for b in range(len(self.B)):
            self.B[b] = dBcache[b]
        
        
    def d_activation_function(self, act : ActivationFunctions, x : np.ndarray):
        match act:
            case ActivationFunctions.RELU:
                return np.array(x > 0, dtype = np.float32)

    def get_cost(self, x : np.ndarray):
        pred = self.make_prediction()
        match self.cost_function:
            case CostFunctions.CROSSENTROPY:
                cost = 0
                for i in range(len(pred)):
                    cost = cost + (-1 * self.train_y[i] * np.log(pred[i]))
                return cost
            case CostFunctions.CLASSCROSSENTROPY:
                return -(1/self.train_x.shape[0])*np.sum(self.train_y * np.log(pred) + (1-self.train_y) * np.log(1-pred))
            case CostFunctions.MSE:
                return (1/(2*self.train_x.shape[0])) * np.sum(pred - self.train_y)
        
    def algorithm(self):
        self.initializing()
        for _ in range(self.iteration):
            self.forward_propagation(self.train_x)
            self.cost_history.append(self.get_cost(self.make_prediction()))
            self.backward_propagation()

    def activation_func(self, x : np.ndarray) -> np.ndarray:
        match self.activation_function:
            case ActivationFunctions.RELU:
                return np.maximum(x, 0)
            case ActivationFunctions.SOFTMAX:
                return (np.exp(x))/sum(np.exp(x))
            case _:
                return x
    
    def select_costfunction(self):
        match self.activation_function:
            case ActivationFunctions.SIGMOID:
                return CostFunctions.CROSSENTROPY
            case ActivationFunctions.SOFTMAX:
                return CostFunctions.CLASSCROSSENTROPY
            case _:
                return CostFunctions.MSE
    
    def sigmoid_function(self, x):
        x = np.clip(x, -709, 709)
        return 1/(1+ np.exp(-x))
    
    def soft_max(self, x):
        return (np.exp(x))/sum(np.exp(x))

    def make_prediction(self) -> np.ndarray:
        match self.type:
            case ProblemType.CLASSIFICATION:
                return self.sigmoid_function(self.A.get(list(self.A.keys())[-1]))
            case ProblemType.MULTICLASSIFICATION:
                return self.soft_max(self.A.get(list(self.A.keys())[-1]))
            case ProblemType.REGRESSION:
                return self.A.get(list(self.A.keys())[-1])
    
    def plot_error_function(self):
        """
        This function is used to plot the error function
        """
        plt.plot(range(self.iteration), self.cost_history, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function during gradient descent')
        plt.show()
    