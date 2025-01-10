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

        self.W = {}
        self.B = {}
        self.A = {}
        self.cost_function = self.select_costfunction()
        self.cost_history = []
        self.output_size = 0
        self.type = self.type_problem()
        self.Z_cache = {}
        
        
    def type_problem(self) -> ProblemType:
        match self.cost_function:
            case CostFunctions.CROSSENTROPY:
                self.output_size = 1
                return ProblemType.CLASSIFICATION
            case CostFunctions.CLASSCROSSENTROPY:
                self.output_size = self.test_y.shape[0]
                return ProblemType.MULTICLASSIFICATION
            case CostFunctions.MSE:
                self.output_size = 1
                return ProblemType.REGRESSION
    
    
    def initializing(self):
        np.random.seed(42)
        random.seed(42)
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
        for layer in range(self.hidden_layers + 1):
            if layer == 0:
                Z = np.dot(self.W.get(layer), input) + self.B[layer]
                
                self.Z_cache[layer] = Z

                self.A[layer] = self.relu(Z)
            elif layer == self.hidden_layers:
                Z = np.dot(self.W.get(layer), self.A.get(layer - 1)) + self.B[layer]
                self.Z_cache[layer] = Z
                self.A[layer] = self.soft_max(Z)
                
            else:
                Z = np.dot(self.W.get(layer), self.A.get(layer - 1)) + self.B[layer]
                self.Z_cache[layer] = Z
                self.A[layer] = self.relu(Z)


    def backward_propagation(self):
        dZcache = {}
        dWcache = {}
        dBcache = {}
        
        for layer in range(len(self.A) - 1, -1, -1):
            if layer == len(self.A) - 1:
                dZ = self.A.get(layer) - self.train_y
                dZcache[layer] = dZ
                dW = 1/self.train_y.shape[1] * np.dot(dZ, np.transpose(self.A.get(layer - 1)))
                dWcache[layer] = dW
                dB = 1/self.train_y.shape[1] * np.sum(dZ, 1).reshape(-1,1)
                dBcache[layer] = dB
            elif layer == 0:
                dZ = np.dot(np.transpose(self.W.get(layer + 1)), dZcache.get(layer + 1)) * self.d_activation_function(ActivationFunctions.RELU, self.Z_cache.get(layer))
                dZcache[layer] = dZ
                dW = 1/self.train_y.shape[1] * np.dot(dZ, self.train_x.T)
                dWcache[layer] = dW
                dB = 1/self.train_y.shape[1] * np.sum(dZ, 1).reshape(-1,1)
                dBcache[layer] = dB
            else:
                dZ = np.dot(np.transpose(self.W.get(layer + 1)), dZcache.get(layer + 1)) * self.d_activation_function(ActivationFunctions.RELU, self.Z_cache.get(layer))
                dZcache[layer] = dZ
                dW = 1/self.train_y.shape[1] * np.dot(dZ, np.transpose(self.A.get(layer)))
                dWcache[layer] = dW
                dB = 1/self.train_y.shape[1] * np.sum(dZ, 1).reshape(-1,1)
                dBcache[layer] = dB

        for weights in range(len(self.W)):
            self.W[weights] = self.W[weights] - self.learning_rate * dWcache.get(weights)
        
        for b in range(len(self.B)):
            self.B[b] = self.B[b] - self.learning_rate * dBcache.get(b)        
    
    def d_activation_function(self, act : ActivationFunctions, x : np.ndarray):
        match act:
            case ActivationFunctions.RELU:
                return x > 0
            
    def get_cost(self):
        pred = self.make_prediction()
        match self.cost_function:
            case CostFunctions.CROSSENTROPY:
                cost = 0
                for i in range(len(pred)):
                    cost = cost + (-1 * self.train_y[i] * np.log(pred[i]))
                return cost
            
            case CostFunctions.CLASSCROSSENTROPY:
                return -(1/self.train_y.shape[1])*np.sum(self.train_y*np.log(pred))
            
            case CostFunctions.MSE:
                return (1/(2*self.train_y.shape[1])) * np.sum(pred - self.train_y.T)
        
    def algorithm(self):
        self.initializing()
        for _ in range(self.iteration):
            self.forward_propagation(self.train_x)
            cost = self.get_cost()
            self.backward_propagation()
            self.cost_history.append(cost)
            if _ % 100 == 0:
                print('Iteration: ', _)
                print('Accuracy: ', self.get_accuracy(self.get_predictions(self.A.get(self.hidden_layers)), np.argmax(self.train_y, axis=0)))
            
            
    def relu(self, Z : np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    
    def activation_func(self, x : np.ndarray) -> np.ndarray:
        match self.activation_function:
            case ActivationFunctions.RELU:
                return np.maximum(x, 0)
            case ActivationFunctions.SOFTMAX:
                exp_x = np.exp(x)
                return (exp_x)/np.sum(exp_x, keepdims=True)
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
        return 1/(1+ np.exp(-x))
    
    def soft_max(self, x):
        softmax_arr = np.exp(x)/np.sum(np.exp(x), axis = 0)
        return softmax_arr

    def make_prediction(self) -> np.ndarray:
        return self.A.get(self.hidden_layers)

    def predict(self, features : np.ndarray):
        self.forward_propagation(features)
        return self.A.get(self.hidden_layers)
    
    def get_predictions(self, Amax):
        return np.argmax(Amax,0)
    
    def get_accuracy(self, predictions, Y):
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

    def make_new_pred(self):
        self.forward_propagation(self.test_x)
        prediction = self.get_predictions(self.A.get(self.hidden_layers))
        actual = np.argmax(self.test_y, axis=0)
        df = pd.DataFrame({
            'prediction' : prediction,
            'actual' : actual
        })
        print('Accuracy: ', self.get_accuracy(prediction, actual))
        print('\n')
        print('Testing matrix of predicted valued vs actual values')
        pivot = pd.crosstab (df['prediction'], df['actual'])
        print(pivot)
        